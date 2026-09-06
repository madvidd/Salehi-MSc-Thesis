"""CPU regression tests. The Mamba CUDA kernel is verified by the lab preflight."""
import ast
import importlib.util
import json
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

from common import (atomic_json, checkpoint_metric, choose_epoch_checkpoint,
                    read_json, source_inventory, source_sha256, token_from_file)
from publish_combined import create_stage, preserve_terminal, validate_stage
from setup_combined import combine_source

PACKAGE = Path(__file__).resolve().parent
OLD = PACKAGE.parent / "SEAM_AV2_Mamba_3_Run"
CONTROLS = PACKAGE.parent / "SEAM_20_Epoch_4_Test_Suite"


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_protocol_unchanged(self):
        p = read_json(PACKAGE / "protocol.json")
        self.assertEqual(p["epochs"], 80)
        self.assertEqual(p["devices"] * p["microbatch"] * p["accumulation"], 32)
        self.assertEqual(p["dataset"]["split_points"], [30, 40, 50])
        self.assertEqual(round(p["epochs"] * p["warmup_ratio"]), 13)

    def test_token_encodings_and_ambiguity(self):
        token = "ghp_" + "a" * 36
        path = self.root / "secret.txt"
        for encoding in ("utf-8-sig", "utf-16"):
            path.write_text(token, encoding=encoding)
            self.assertEqual(token_from_file(path), token)
        path.write_text(token + "\nghp_" + "b" * 36)
        with self.assertRaises(RuntimeError):
            token_from_file(path)

    def test_append_preserves_rotation_and_existing_lines(self):
        src, dest = self.root / "suite.log", self.root / "Terminal.txt"
        src.write_bytes(b"first\n")
        preserve_terminal(src, dest)
        with src.open("ab") as stream:
            stream.write(b"second\n")
        preserve_terminal(src, dest)
        preserve_terminal(src, dest)
        self.assertEqual(dest.read_bytes(), b"first\nsecond\n")
        src.write_bytes(b"new\n")
        preserve_terminal(src, dest)
        self.assertTrue(dest.read_bytes().startswith(b"first\nsecond\n"))
        self.assertTrue(dest.read_bytes().endswith(b"new\n"))

    def test_corrupt_checkpoint_fallback_and_no_deletion(self):
        good = self.root / "epoch_001-minADE6_1.00000000.ckpt"
        bad = self.root / "epoch_002-minADE6_0.90000000.ckpt"
        good.write_text("good")
        bad.write_text("corrupted")
        def load(path):
            if path == bad:
                raise ValueError("corrupt")
            return {"state_dict": {"a": 1}, "optimizer_states": [1], "lr_schedulers": [1],
                    "combined_protocol": "seam80_combined_v1", "epoch": 1}
        path, state, failures = choose_epoch_checkpoint(self.root, load)
        self.assertEqual(path, good)
        self.assertEqual(state["epoch"], 1)
        self.assertEqual(len(failures), 1)
        self.assertTrue(bad.exists())
        self.assertEqual(checkpoint_metric("epoch_079-minADE6_0.60000000-v1.ckpt"), 0.6)

    def test_size_and_secret_gates(self):
        (self.root / "Summary.md").write_text("safe\n")
        self.assertEqual(validate_stage(self.root), ["Summary.md"])
        token_file = self.root / "bad.txt"
        token_file.write_text("ghp_" + "a" * 36)
        with self.assertRaises(RuntimeError):
            validate_stage(self.root)
        token_file.unlink()
        large = self.root / "large.txt"
        with large.open("wb") as stream:
            stream.truncate(10 * 1024**2)
        with self.assertRaises(RuntimeError):
            validate_stage(self.root)

    def test_cross_platform_hash(self):
        a, b = self.root / "a.py", self.root / "b.py"
        a.write_bytes(b"x=1\r\n")
        b.write_bytes(b"x=1\n")
        self.assertEqual(source_sha256(a), source_sha256(b))

    def test_input_hashes_match(self):
        for relative, expected in read_json(PACKAGE / "INPUT_HASHES.json").items():
            self.assertEqual(source_sha256(PACKAGE.parent / relative), expected, relative)

    def test_setup_is_isolated_and_idempotent(self):
        from setup_combined import main
        from common import assert_sources
        previous = self.root / "Results/SEAM_AV2_MAMBA_3RUN_20260824-202936"
        previous.mkdir(parents=True)
        preserved = previous / "KEEP.txt"
        preserved.write_text("previous results")
        data = self.root / "data"
        for split in ("train", "val"):
            (data / split).mkdir(parents=True)
            (data / split / "fixture.pt").write_bytes(b"path-only test fixture")
        (previous / "DATA_ROOT.txt").write_text(str(data))
        with patch.object(sys, "argv", ["setup", "--base", str(self.root)]):
            with patch("setup_combined.shutil.disk_usage", return_value=types.SimpleNamespace(free=100 * 1024**3)):
                main()
                pointer = self.root / "Codes/LATEST_SEAM_AV2_80EPOCH_COMBINED_CODE.txt"
                first = pointer.read_text()
                assert_sources(Path(first.strip()))
                main()
                self.assertEqual(pointer.read_text(), first)
        self.assertEqual(preserved.read_text(), "previous results")

    def test_snapshot_limits_and_all_six_metrics(self):
        experiment, results = self.root / "experiment", self.root / "results"
        (experiment / "reference_config").mkdir(parents=True)
        run = results / "combined"
        run.mkdir(parents=True)
        manifest = {"base": str(self.root), "results": str(results), "experiment": str(experiment)}
        atomic_json(experiment / "RUN_MANIFEST.json", manifest)
        code = experiment / "Code"
        shutil.copytree(CONTROLS / "upstream/seam-main", code,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        combine_source(code, OLD / "upstream/seam-main/src/model/layers/mamba_layers.py", PACKAGE)
        (results / "suite.log").write_bytes(b"progress\n" * 1000000)
        (run / "epoch_metrics.csv").write_text(
            "epoch,global_step,epoch_seconds,MR,minADE1,minADE6,minFDE1,minFDE6,b-minFDE6\n"
            "0,100,60,0.2,1.2,0.7,3.0,1.3,1.9\n", encoding="utf-8")
        stage = self.root / "stage"
        create_stage(experiment, stage)
        files = validate_stage(stage)
        self.assertIn("Terminal.txt", files)
        self.assertLess((stage / "Terminal.txt").stat().st_size, 8 * 1024**2)
        self.assertIn("Best validation epoch 0", (stage / "Summary.md").read_text())
        self.assertIn("1.900000", (stage / "Summary.md").read_text())

    def test_syntax_and_combined_source_generation(self):
        code = self.root / "Code"
        shutil.copytree(CONTROLS / "upstream/seam-main", code,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        combine_source(code, OLD / "upstream/seam-main/src/model/layers/mamba_layers.py", PACKAGE)
        for path in [*code.rglob("*.py"), *PACKAGE.glob("*.py")]:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.assertIn("'combined'", (code / "src/model/seam.py").read_text())


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch not installed")
class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.code = Path(cls.temp.name) / "Code"
        shutil.copytree(CONTROLS / "upstream/seam-main", cls.code,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        combine_source(cls.code, OLD / "upstream/seam-main/src/model/layers/mamba_layers.py", PACKAGE)
        sys.path.insert(0, str(cls.code))
        import torch
        torch.set_num_threads(2)
        from src.model.layers.controlled_ablation import QKNormMultiheadAttention, RelativeGeometryBias
        cls.attention = QKNormMultiheadAttention
        cls.geometry = RelativeGeometryBias

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.code))
        cls.temp.cleanup()

    def test_qknorm_geometry_mask_finite_backward(self):
        import torch
        q = torch.randn(2, 5, 128, requires_grad=True)
        attn = self.attention(128, 8, dropout=0, bias=False)
        geometry = self.geometry(8)
        valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
        bias = geometry(torch.randn(2, 5, 2), torch.randn(2, 5), valid)
        out, weights = attn(q, q, q, attn_mask=bias)
        self.assertEqual(tuple(out.shape), (2, 5, 128))
        self.assertEqual(float(weights[0, :, 3:].abs().sum()), 0.0)
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(geometry.network[-1].weight.grad).all())

    def test_fully_padded_rows_zero_and_finite(self):
        import torch
        q = torch.randn(2, 3, 16, requires_grad=True)
        attn = self.attention(16, 4, dropout=0, bias=False)
        out, weights = attn(q, q, q, key_padding_mask=torch.ones(2, 3, dtype=torch.bool))
        self.assertEqual(float(out.abs().sum()), 0)
        self.assertEqual(float(weights.abs().sum()), 0)
        out.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_combined_streamed_forward_and_optimizer_wiring(self):
        import torch
        from torch import nn
        # Structural substitute only: production uses the actual pinned CUDA
        # Mamba implementation, tested by the mandatory lab DDP preflight.
        class ShapeOnlyMamba(nn.Module):
            def __init__(self, d_model, d_state, d_conv, expand, use_fast_path):
                super().__init__()
                self.d_state, self.d_conv, self.expand = d_state, d_conv, expand
                self.in_proj = nn.Linear(d_model, d_model)
                self.dt_proj = nn.Linear(d_model, d_model)
                nn.init.ones_(self.dt_proj.bias)
            def forward(self, x):
                return self.in_proj(x) + 0.001 * self.dt_proj(x)
        stub = types.ModuleType("mamba_ssm")
        stub.Mamba = ShapeOnlyMamba
        sys.modules["mamba_ssm"] = stub
        from src.model.combined import CombinedSeam
        model = CombinedSeam(embed_dim=128, encoder_depth=4, num_heads=8, mlp_ratio=4.0,
                             qkv_bias=False, drop_path=0.2, future_steps=80, k=6, ma=False)
        memory, loss = None, None
        for t in (3., 4., 5.):
            frame = {
                "x_valid_mask": torch.ones(2, 3, 30, dtype=torch.bool),
                "x_key_valid_mask": torch.ones(2, 3, dtype=torch.bool),
                "x_positions_diff": torch.randn(2, 3, 30, 2),
                "x_velocity_diff": torch.randn(2, 3, 30),
                "lane_valid_mask": torch.ones(2, 4, 20, dtype=torch.bool),
                "lane_key_valid_mask": torch.ones(2, 4, dtype=torch.bool),
                "lane_positions": torch.randn(2, 4, 20, 2),
                "lane_centers": torch.randn(2, 4, 2), "lane_angles": torch.randn(2, 4),
                "x_centers": torch.randn(2, 3, 2), "x_angles": torch.randn(2, 3, 30),
                "x_attr": torch.zeros(2, 3, 3), "origin": torch.zeros(2, 2),
                "theta": torch.zeros(2), "timestamp": torch.full((2,), t), "memory_dict": memory,
            }
            output = model(frame)
            self.assertEqual(tuple(output["y_hat"].shape), (2, 6, 80, 2))
            term = output["y_hat"].square().mean() + output["pi"].square().mean()
            loss = term if loss is None else loss + term
            memory = output["memory_dict"]
            self.assertIn("pi", memory)
        loss.backward()
        for name in ("uncertainty_target_context.controller.weight", "relative_geometry_bias.network.2.weight",
                     "h_embed.0.attn.logit_scale", "decoder.loc.blocks.0.mixer.in_proj.weight"):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(sum(isinstance(m, self.attention) for m in model.modules()), 20)
        from src.model.pl_modules import StreamLightningModule
        from types import SimpleNamespace
        module = StreamLightningModule(model=model, ma=False, num_grad_frame=3,
                    optim=SimpleNamespace(lr=.001, min_lr=.00001, weight_decay=.01, warmup_ratio=.167, epochs=80))
        from train_combined import audit_model
        audit_model(module)
        # Compose the actual reference YAML files, not a hand-written stand-in.
        from train_combined import build
        reference = Path(self.temp.name) / "reference_config"
        shutil.copytree(OLD / "Results/SEAM_AV2_MAMBA_3RUN_20260824-202936/baseline/config", reference)
        compiled, datamodule = build(Path(self.temp.name), {"protocol": read_json(PACKAGE / "protocol.json"),
                    "data_root": "/read-only-test-dataset", "workers_per_rank": 4})
        self.assertEqual(datamodule.batch_size, 8)
        self.assertEqual(datamodule.test_batch_size, 16)
        self.assertEqual(compiled.optim.epochs, 80)
        self.assertEqual(compiled.num_grad_frame, 3)

        # Exercise the production Lightning callbacks and resume path on CPU.
        # Synthetic scenes and the shape-only mixer do not stand in for AV2/CUDA.
        import copy
        import pytorch_lightning as pl
        from pytorch_lightning.loggers import CSVLogger
        import train_combined
        template = {key: value for key, value in frame.items() if key != "memory_dict"}
        def collate(_):
            frames = []
            for index, timestamp in enumerate((3., 4., 5.)):
                sample = copy.deepcopy(template)
                sample.update(timestamp=torch.full((2,), timestamp), scenario_id=["a", "b"],
                              target=torch.randn(2, 3, 80 - index * 10, 2),
                              target_mask=torch.ones(2, 3, 80 - index * 10, dtype=torch.bool))
                frames.append(sample)
            return frames
        class SyntheticData(pl.LightningDataModule):
            def train_dataloader(self):
                return torch.utils.data.DataLoader(range(4), batch_size=1, collate_fn=collate)
            def val_dataloader(self):
                return torch.utils.data.DataLoader(range(2), batch_size=1, collate_fn=collate)
        original_trainer = pl.Trainer
        def cpu_trainer(**kwargs):
            kwargs.update(accelerator="cpu", devices=1, strategy="auto", sync_batchnorm=False, enable_model_summary=False)
            return original_trainer(**kwargs)
        result_root = Path(self.temp.name) / "results"
        manifest = {"results": str(result_root), "cpu_threads_per_rank": 2,
                    "protocol": read_json(PACKAGE / "protocol.json")}
        atomic_json(Path(self.temp.name) / "RUN_MANIFEST.json", manifest)
        with patch.object(train_combined, "environment_check", return_value={}), \
             patch.object(train_combined, "assert_sources"), \
             patch.object(train_combined, "build", return_value=(compiled, SyntheticData())), \
             patch.object(pl, "Trainer", side_effect=cpu_trainer), \
             patch("pytorch_lightning.loggers.TensorBoardLogger", CSVLogger):
            for phase in ("smoke", "smoke_resume"):
                with patch.object(sys, "argv", ["train", "--experiment", self.temp.name, "--phase", phase]):
                    train_combined.main()
        restored = read_json(result_root / "preflight/smoke_resume_OK.json")
        self.assertGreater(restored["global_step"], restored["previous_step"])
        import csv
        with (result_root / "preflight/epoch_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["epoch"]) for row in rows], [0, 1])
        checkpoint = torch.load(result_root / "preflight/checkpoints/last.ckpt", weights_only=False)
        self.assertEqual(checkpoint["global_step"], 4)
        self.assertTrue(checkpoint["optimizer_states"][0]["state"])
        self.assertEqual(checkpoint["lr_schedulers"][0]["epochs"], 80)


if __name__ == "__main__":
    unittest.main(verbosity=2)
