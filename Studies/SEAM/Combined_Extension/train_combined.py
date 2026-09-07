import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from common import METRICS, assert_sources, atomic_json, checkpoint_metric, choose_epoch_checkpoint, read_json


def environment_check():
    from importlib.metadata import version
    import torch
    expected = {
        "pytorch-lightning": "2.4.0", "torchmetrics": "1.5.0", "timm": "1.0.11",
        "transformers": "4.44.2", "numpy": "1.26.4", "mamba-ssm": "1.2.2",
        "causal-conv1d": "1.2.2.post1", "hydra-core": "1.3.2",
    }
    for name, required in expected.items():
        observed = version(name)
        if observed != required:
            raise RuntimeError(f"Existing SEAM environment: {name}={observed}; expected {required}")
    if not torch.__version__.startswith("2.1.1") or torch.version.cuda != "12.1":
        raise RuntimeError(f"Expected previous torch 2.1.1/cu121 environment; got {torch.__version__}/{torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no training was started")
    return {**{name: version(name) for name in expected}, "torch": torch.__version__, "cuda": torch.version.cuda}


def build(experiment, manifest):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from src.model.combined import CombinedSeam
    from src.model.pl_modules import StreamLightningModule

    p = manifest["protocol"]
    reference_model = OmegaConf.load(experiment / "reference_config/Seam.yaml").pl_module
    reference_data = OmegaConf.load(experiment / "reference_config/av2_stream.yaml").pl_module
    reference_train = OmegaConf.load(experiment / "reference_config/config.yaml")
    for name, expected in (("lr", p["learning_rate"]), ("min_lr", p["minimum_learning_rate"]),
                           ("weight_decay", p["weight_decay"]), ("warmup_ratio", p["warmup_ratio"])):
        if reference_model.optim[name] != expected:
            raise RuntimeError(f"Previous baseline optimizer mismatch: {name}")
    if reference_train.epochs != p["epochs"] or reference_train.seed != p["seed"]:
        raise RuntimeError("Previous baseline epoch/seed mismatch")
    if OmegaConf.to_container(reference_data.dataset) != p["dataset"]:
        raise RuntimeError("Previous baseline data protocol mismatch")
    if p["devices"] * p["microbatch"] * p["accumulation"] != 32:
        raise RuntimeError("Global batch must remain 32")
    model_cfg = OmegaConf.to_container(reference_model.model, resolve=False)
    for key in ("_target_", "variant", "mamba_d_state", "mamba_d_conv", "mamba_expand", "mamba_future_depth"):
        model_cfg.pop(key, None)
    model = CombinedSeam(**model_cfg)
    optim = OmegaConf.create({
        "lr": p["learning_rate"], "min_lr": p["minimum_learning_rate"],
        "weight_decay": p["weight_decay"], "warmup_ratio": p["warmup_ratio"], "epochs": p["epochs"],
    })
    module = StreamLightningModule(model=model, optim=optim, ma=False, num_grad_frame=p["num_grad_frame"])
    reference_data.data_root = manifest["data_root"]
    reference_data.train_batch_size = p["microbatch"]
    reference_data.test_batch_size = p["validation_batch"]
    reference_data.num_workers = manifest["workers_per_rank"]
    dm = instantiate(reference_data)
    return module, dm


def audit_model(module):
    import torch
    from src.model.layers.controlled_ablation import QKNormMultiheadAttention, RelativeGeometryBias, UncertaintyTargetContext
    from src.model.layers.mamba_layers import MambaTrajectoryHead

    modules = list(module.model.modules())
    observed = {
        "qknorm": sum(isinstance(m, QKNormMultiheadAttention) for m in modules),
        "geometry": sum(isinstance(m, RelativeGeometryBias) for m in modules),
        "uncertainty": sum(isinstance(m, UncertaintyTargetContext) for m in modules),
        "future_head": sum(isinstance(m, MambaTrajectoryHead) for m in modules),
        "plain_mha": sum(isinstance(m, torch.nn.MultiheadAttention) for m in modules),
    }
    if observed != {"qknorm": 20, "geometry": 1, "uncertainty": 1, "future_head": 1, "plain_mha": 0}:
        raise RuntimeError(f"Architecture audit failed: {observed}")
    if len(module.model.decoder.loc.blocks) != 2:
        raise RuntimeError("Future head must retain two Mamba blocks")
    for block in module.model.decoder.loc.blocks:
        mixer = block.mixer
        if mixer.d_state != 16 or mixer.d_conv != 4 or mixer.expand != 2:
            raise RuntimeError("Future-head Mamba configuration changed")
        if float(mixer.dt_proj.bias.abs().sum()) == 0:
            raise RuntimeError("Mamba time-step initialisation was overwritten")
    optimizers, schedulers = module.configure_optimizers()
    params = [id(p) for g in optimizers[0].param_groups for p in g["params"]]
    expected = {id(p) for p in module.parameters() if p.requires_grad}
    if len(params) != len(set(params)) or set(params) != expected or len(schedulers) != 1:
        raise RuntimeError("Optimizer parameter coverage/uniqueness failed")
    print("COMBINED_ARCHITECTURE_OK=" + json.dumps(observed), flush=True)
    return {**observed, "parameters": sum(p.numel() for p in module.parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--phase", choices=("probe", "smoke", "smoke_resume", "train", "evaluate"), required=True)
    args = parser.parse_args()
    experiment = Path(args.experiment).resolve()
    assert_sources(experiment)
    manifest = read_json(experiment / "RUN_MANIFEST.json")
    results = Path(manifest["results"])
    phase, p = args.phase, manifest["protocol"]
    sys.path.insert(0, str(experiment / "Code"))
    versions = environment_check()

    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy

    torch.set_num_threads(manifest["cpu_threads_per_rank"])
    pl.seed_everything(p["seed"], workers=True)
    module, dm = build(experiment, manifest)
    architecture = audit_model(module)
    if phase == "probe":
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Two free training GPUs are required")
        for index in range(2):
            free, total = torch.cuda.mem_get_info(index)
            if free < 7 * 1024**3:
                raise RuntimeError(f"GPU {index} has less than 7 GiB free. Existing jobs were not touched.")
            print(f"GPU_{index}={torch.cuda.get_device_name(index)} free={free} total={total}")
        atomic_json(results / "ENVIRONMENT.json", versions)
        atomic_json(results / "ARCHITECTURE_AUDIT.json", architecture)
        return

    smoke = phase in {"smoke", "smoke_resume"}
    output = results / "preflight" if smoke else results / "combined"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, state, rejected = choose_epoch_checkpoint(
        output / "checkpoints", lambda f: torch.load(f, map_location="cpu", weights_only=False))
    initial_step = state["global_step"] if state else -1
    initial_epoch = state["epoch"] if state else -1
    if phase == "smoke_resume" and checkpoint is None:
        raise RuntimeError("Preflight resume checkpoint is missing")
    if state and state.get("run_root") != str(results):
        raise RuntimeError("Refusing to resume a checkpoint from another experiment")
    if state and state.get("checkpoint_family") != ("preflight" if smoke else "training"):
        raise RuntimeError("Preflight and full-training checkpoints cannot be interchanged")
    del state
    if os.environ.get("RANK", "0") == "0":
        atomic_json(output / "RESUME.json", {"checkpoint": str(checkpoint) if checkpoint else None,
                     "epoch": initial_epoch, "global_step": initial_step, "rejected": rejected})

    class Evidence(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            self.started = time.monotonic()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.is_global_zero and (batch_idx % 50 == 0):
                progress = {"epoch": trainer.current_epoch, "global_step": trainer.global_step,
                            "batch": batch_idx, "batches": trainer.num_training_batches,
                            "updated_unix": time.time(), "phase": phase}
                atomic_json(output / "HEARTBEAT.json", progress)
                print("PROGRESS=" + json.dumps(progress), flush=True)

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            finite = torch.ones((), device=pl_module.device, dtype=torch.int32)
            for parameter in pl_module.parameters():
                if parameter.grad is not None:
                    finite *= torch.isfinite(parameter.grad).all().to(torch.int32)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError("Non-finite gradient detected; last completed epoch was retained")
            if smoke:
                for name in ("uncertainty_target_context.controller.weight", "relative_geometry_bias.network.2.weight",
                             "h_embed.0.attn.logit_scale", "decoder.loc.blocks.0.mixer.in_proj.weight"):
                    parameter = dict(pl_module.model.named_parameters())[name]
                    if parameter.grad is None:
                        raise RuntimeError(f"Required intervention gradient missing: {name}")

        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking or not trainer.is_global_zero:
                return
            metrics = {key: float(trainer.callback_metrics[key].detach().cpu()) for key in METRICS}
            if not all(math.isfinite(value) for value in metrics.values()):
                raise RuntimeError("Non-finite validation metric")
            row = {"epoch": trainer.current_epoch, "global_step": trainer.global_step,
                   "epoch_seconds": round(time.monotonic() - getattr(self, "started", time.monotonic()), 3), **metrics}
            target = output / "epoch_metrics.csv"
            exists = target.exists()
            with target.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
            print("VALIDATION=" + json.dumps(row), flush=True)

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint["combined_protocol"] = "seam80_combined_v1"
            checkpoint["run_root"] = str(results)
            checkpoint["checkpoint_family"] = "preflight" if smoke else "training"

    # Every completed epoch retains a full optimiser/scheduler/loop checkpoint.
    # Default Lightning TorchCheckpointIO performs an atomic local-file save.
    saver = ModelCheckpoint(dirpath=output / "checkpoints", filename="epoch_{epoch:03d}-minADE6_{minADE6:.8f}",
                            monitor="minADE6", mode="min", save_top_k=-1, save_last=True,
                            every_n_epochs=1, save_on_train_epoch_end=True, auto_insert_metric_name=False)
    evaluate = phase == "evaluate"
    version = os.environ.get("SEAM_ATTEMPT_ID", phase)
    trainer = pl.Trainer(
        accelerator="gpu", devices=1 if evaluate else 2,
        strategy="auto" if evaluate else DDPStrategy(find_unused_parameters=False, timeout=timedelta(minutes=10)),
        max_epochs=(initial_epoch + 2 if phase == "smoke_resume" else 1) if smoke else p["epochs"],
        precision=p["precision"], sync_batchnorm=False if evaluate else p["sync_batchnorm"],
        accumulate_grad_batches=p["accumulation"], gradient_clip_val=p["gradient_clip_norm"],
        gradient_clip_algorithm="norm", callbacks=[] if evaluate else [Evidence(), saver],
        enable_checkpointing=not evaluate,
        logger=[CSVLogger(str(output / "logs"), name="csv", version=version),
                TensorBoardLogger(str(output / "logs"), name="tensorboard", version=version)] if not evaluate else False,
        enable_progress_bar=False, log_every_n_steps=1 if smoke else 50,
        limit_train_batches=4 if smoke else 1.0, limit_val_batches=2 if smoke else 1.0,
        num_sanity_val_steps=2, default_root_dir=str(output),
    )
    if evaluate:
        with (output / "epoch_metrics.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        best = min(rows, key=lambda r: float(r["minADE6"]))
        matches = sorted((output / "checkpoints").glob(f"epoch_{int(best['epoch']):03d}-minADE6_*.ckpt"))
        if not matches:
            raise RuntimeError("Selected minADE6 checkpoint missing")
        best_path = min(matches, key=lambda f: abs(checkpoint_metric(f) - float(best["minADE6"])))
        metrics = trainer.validate(module, datamodule=dm, ckpt_path=str(best_path), verbose=True)[0]
        values = {key: float(metrics[key]) for key in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError("Non-finite final metric")
        atomic_json(output / "FINAL_METRICS.json", {"checkpoint": str(best_path), "selected_epoch": int(best["epoch"]),
                    "evaluation": "one GPU, complete validation split, batch 16", "metrics": values})
    else:
        trainer.fit(module, datamodule=dm, ckpt_path=str(checkpoint) if checkpoint else None)
        if trainer.is_global_zero:
            if trainer.global_step <= initial_step and initial_epoch < p["epochs"] - 1:
                raise RuntimeError("Training did not advance from the restored checkpoint")
            if smoke:
                atomic_json(output / f"{phase}_OK.json", {"previous_step": initial_step,
                    "global_step": trainer.global_step, "completed_epochs": trainer.current_epoch})
            elif trainer.current_epoch >= p["epochs"] and list((output / "checkpoints").glob("epoch_079-*.ckpt")):
                atomic_json(output / "TRAINING_COMPLETE.json", {"epochs": trainer.current_epoch, "step": trainer.global_step})
            else:
                raise RuntimeError("Training exited without completing 80 epochs")


if __name__ == "__main__":
    main()
