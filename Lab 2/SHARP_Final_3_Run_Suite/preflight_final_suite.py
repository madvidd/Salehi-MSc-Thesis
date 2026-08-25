#!/usr/bin/env python3
"""Static, CUDA, optimizer, DDP, and real-batch preflight for all three runs."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


VARIANTS = (
    ("01_official_sharp_baseline", "official_sharp_baseline"),
    ("02_qknorm_uncertainty_geometry", "qknorm_uncertainty_geometry"),
    (
        "03_qknorm_uncertainty_geometry_temporal_mamba",
        "qknorm_uncertainty_geometry_temporal_mamba",
    ),
)
FORBIDDEN = (
    "Traceback (most recent call last)",
    "Error executing job with overrides",
    "illegal memory access",
    "CUDA out of memory",
    "NCCL WARN",
    "Trying to infer the `batch_size`",
    "Support for mismatched key_padding_mask and attn_mask",
)


def run_checked(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log: Path,
    timeout_seconds: int = 900,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            returncode = 124
            handle.write(
                f"\nPREFLIGHT_TIMEOUT={timeout_seconds}s command={command!r}\n"
            )
        if returncode:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    text = log.read_text(encoding="utf-8", errors="replace")
    failures = [pattern for pattern in FORBIDDEN if pattern.lower() in text.lower()]
    if returncode or failures:
        tail = "\n".join(text.replace("\r", "\n").splitlines()[-120:])
        raise RuntimeError(
            f"Preflight failed ({returncode}) in {log}; "
            f"forbidden={failures}\n{tail}"
        )


def nccl_smoke(
    python: Path,
    experiment: Path,
    results: Path,
    environment: dict[str, str],
) -> None:
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        str(experiment / "nccl_collective_preflight.py"),
    ]
    normal = environment.copy()
    normal.pop("NCCL_P2P_DISABLE", None)
    try:
        run_checked(
            command,
            experiment,
            normal,
            results / "preflight/nccl_four_gpu.log",
            timeout_seconds=300,
        )
        p2p_disable = "0"
        mode = "native-p2p"
    except RuntimeError as first_error:
        fallback = environment.copy()
        fallback["NCCL_P2P_DISABLE"] = "1"
        run_checked(
            command,
            experiment,
            fallback,
            results / "preflight/nccl_four_gpu_no_p2p.log",
            timeout_seconds=300,
        )
        p2p_disable = "1"
        mode = "shared-memory-fallback"
        (results / "preflight/nccl_native_p2p_failure.txt").write_text(
            str(first_error) + "\n", encoding="utf-8"
        )
    (results / "NCCL_RUNTIME.env").write_text(
        f"export NCCL_P2P_DISABLE={p2p_disable}\n"
        "export NCCL_IB_DISABLE=1\n"
        "export TORCH_NCCL_ASYNC_ERROR_HANDLING=1\n"
        "export TORCH_NCCL_BLOCKING_WAIT=1\n"
        "export TORCH_NCCL_DUMP_ON_TIMEOUT=1\n"
        "export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576\n",
        encoding="utf-8",
    )
    print(f"FOUR_GPU_NCCL_PREFLIGHT_OK={mode}")


def static_audit(experiment: Path) -> None:
    for slug, variant in VARIANTS:
        code = experiment / "variants" / slug / "Code"
        sharp = (code / "src/model/sharp.py").read_text(encoding="utf-8")
        pl_module = (code / "src/model/pl_modules.py").read_text(encoding="utf-8")
        config = (code / "conf/config.yaml").read_text(encoding="utf-8")
        model_config = (code / "conf/model/Sharp_av2.yaml").read_text(
            encoding="utf-8"
        )
        data_config = (code / "conf/datamodule/av2_stream.yaml").read_text(
            encoding="utf-8"
        )
        required = (
            "gpus: 4",
            "batch_size: 8",
            "epochs: 80",
            "save_last: True",
            "precision: 32-true",
            "sync_batchnorm: true",
        )
        missing = [value for value in required if value not in config]
        if missing:
            raise RuntimeError(f"{slug} config audit failed: {missing}")
        if "RichProgressBar" in config or "TQDMProgressBar" not in config:
            raise RuntimeError(f"{slug} progress callback audit failed")
        for value in ("lr: 1e-4", "min_lr: 1e-5", "warmup_ratio: 0.1625"):
            if value not in model_config:
                raise RuntimeError(f"{slug} model config is missing {value}")
        architecture = (
            "embed_dim: 128",
            "future_steps: 100",
            "encoder_depth: 4",
            "num_heads: 8",
            "mlp_ratio: 4.0",
            "qkv_bias: False",
            "drop_path: 0.2",
            "use_stream_encoder: True",
            "use_stream_decoder: True",
            "use_target_context: True",
            "dual: True",
            "biased_interaction: True",
            "dm: av2",
            "k: 6",
        )
        missing_architecture = [
            value for value in architecture if value not in model_config
        ]
        if missing_architecture:
            raise RuntimeError(
                f"{slug} official architecture audit failed: {missing_architecture}"
            )
        dataset_contract = (
            "num_historical_steps: 10",
            "split_points: [10, 20, 30, 40, 50]",
            "radius: 150.0",
            "ma: False",
            "num_future_steps: 100",
        )
        missing_dataset = [value for value in dataset_contract if value not in data_config]
        if missing_dataset:
            raise RuntimeError(
                f"{slug} official AV2 data audit failed: {missing_dataset}"
            )
        if "torch.inverse(rot_mat)" in sharp or "rot_mat.transpose(1, 2)" not in sharp:
            raise RuntimeError(f"{slug} rotation stability audit failed")
        if pl_module.count("device=y_hat.device") != 2:
            raise RuntimeError(f"{slug} CUDA index audit failed")
        explicit_stream_batch_sizes = pl_module.count(
            'batch_size=len(data[-1]["scenario_id"])'
        )
        if explicit_stream_batch_sizes != 3:
            raise RuntimeError(
                f"{slug} stream logging batch-size audit failed: "
                f"{explicit_stream_batch_sizes} != 3"
            )
        if "missing_parameters = param_dict.keys() - union_params" not in pl_module:
            raise RuntimeError(f"{slug} optimizer audit is absent")
        qknorm_count = sum(
            path.read_text(encoding="utf-8").count("QKNormMultiheadAttention(")
            for path in (
                code / "src/model/layers/custom_transformer_blocks.py",
                code / "src/model/layers/transformer_blocks.py",
            )
        )
        expected_qknorm = 0 if variant == "official_sharp_baseline" else 3
        if qknorm_count != expected_qknorm:
            raise RuntimeError(
                f"{slug} QKNorm constructor count {qknorm_count} != {expected_qknorm}"
            )
        has_mamba = "TemporalAgentMamba(" in sharp
        if has_mamba != variant.endswith("temporal_mamba"):
            raise RuntimeError(f"{slug} residual-Mamba architecture audit failed")


def qknorm_smoke(
    python: Path,
    code: Path,
    environment: dict[str, str],
    log: Path,
) -> None:
    program = r'''
import torch
from src.model.layers.qknorm_attention import QKNormMultiheadAttention

device = torch.device("cuda:0")
module = QKNormMultiheadAttention(128, 8, dropout=0.2, batch_first=True).to(device)
x = torch.randn(3, 17, 128, device=device, requires_grad=True)
padding = torch.zeros(3, 17, dtype=torch.bool, device=device)
padding[1, 13:] = True
bias = torch.zeros(3 * 8, 17, 17, device=device)
y, _ = module(x, x, x, key_padding_mask=padding, attn_mask=bias)
if y.shape != x.shape or not torch.isfinite(y).all():
    raise SystemExit("QKNorm output audit failed")
y.square().mean().backward()
torch.cuda.synchronize()
if not all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()):
    raise SystemExit("QKNorm gradient audit failed")
print("QKNORM_CUDA_SMOKE_OK")
'''
    run_checked([str(python), "-c", program], code, environment, log)


def mamba_smoke(
    python: Path,
    code: Path,
    environment: dict[str, str],
    log: Path,
) -> None:
    program = r'''
import torch
from src.model.layers.temporal_agent_mamba import (
    TemporalAgentMamba,
    backend_summary,
    stable_cuda_available,
)

if not stable_cuda_available():
    raise SystemExit("Fused selective-scan CUDA backend is unavailable")
device = torch.device("cuda:0")
module = TemporalAgentMamba(128).to(device)
x = torch.randn(257, 10, 128, device=device, requires_grad=True)
valid = torch.ones(257, 10, dtype=torch.bool, device=device)
valid[::3, :4] = False
valid[1::3, :7] = False
y = module(x, valid)
if y.shape != x.shape or not torch.isfinite(y).all():
    raise SystemExit("Residual Mamba output audit failed")
y.square().mean().backward()
torch.cuda.synchronize()
if not all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()):
    raise SystemExit("Residual Mamba gradient audit failed")
print(backend_summary())
print("TEMPORAL_AGENT_MAMBA_CUDA_SMOKE_OK")
'''
    run_checked([str(python), "-c", program], code, environment, log)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--runtime-patch", required=True, type=Path)
    parser.add_argument("--mamba-packages", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    results = args.results.resolve()
    preflight = results / "preflight"
    static_audit(experiment)

    base_environment = os.environ.copy()
    base_environment["PYTHONUNBUFFERED"] = "1"
    base_environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    base_environment["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    base_environment["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    base_environment["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
    base_environment["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1048576"
    base_environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(args.runtime_patch.resolve()),
            str(args.mamba_packages.resolve()),
            base_environment.get("PYTHONPATH", ""),
        )
    )

    qk_code = experiment / "variants/02_qknorm_uncertainty_geometry/Code"
    qk_environment = base_environment.copy()
    qk_environment["PYTHONPATH"] = str(qk_code) + os.pathsep + qk_environment["PYTHONPATH"]
    qknorm_smoke(args.python, qk_code, qk_environment, preflight / "qknorm_cuda.log")

    mamba_code = experiment / "variants/03_qknorm_uncertainty_geometry_temporal_mamba/Code"
    mamba_environment = base_environment.copy()
    mamba_environment["PYTHONPATH"] = str(mamba_code) + os.pathsep + mamba_environment["PYTHONPATH"]
    mamba_smoke(args.python, mamba_code, mamba_environment, preflight / "mamba_cuda.log")

    nccl_smoke(args.python, experiment, results, base_environment)

    for slug, variant in VARIANTS:
        code = experiment / "variants" / slug / "Code"
        output = preflight / slug
        environment = base_environment.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["SHARP_FINAL_VARIANT"] = variant
        environment["PYTHONPATH"] = str(code) + os.pathsep + environment["PYTHONPATH"]
        command = [
            str(args.python),
            "train.py",
            "seed=2333",
            "gpus=1",
            "batch_size=8",
            "epochs=1",
            f"output_dir={output}",
            f"datamodule.pl_module.data_root={args.dataset}",
            "datamodule.pl_module.num_workers=4",
            "trainer.devices=1",
            "trainer.strategy=auto",
            "trainer.sync_batchnorm=false",
            "trainer.precision=32-true",
            "+trainer.fast_dev_run=true",
            "+trainer.num_sanity_val_steps=0",
        ]
        run_checked(
            command,
            code,
            environment,
            output / "preflight.log",
            timeout_seconds=900,
        )
        print(f"REAL_BATCH_SINGLE_GPU_PREFLIGHT_OK={slug}")

    (results / "PREFLIGHT_COMPLETE").write_text(
        "All static, CUDA, optimizer, bounded four-GPU NCCL, and real-batch checks passed.\n",
        encoding="utf-8",
    )
    print("FINAL_SUITE_PREFLIGHT_COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FINAL_SUITE_PREFLIGHT_FAILED: {error}", file=sys.stderr)
        raise
