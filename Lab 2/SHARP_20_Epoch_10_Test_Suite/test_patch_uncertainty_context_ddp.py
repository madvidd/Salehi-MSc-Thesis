#!/usr/bin/env python3
"""Regression test for the idempotent uncertainty-context recovery patch."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path


PATCH_PATH = Path(__file__).with_name("patch_uncertainty_context_ddp.py")
SPEC = importlib.util.spec_from_file_location("uncertainty_recovery_patch", PATCH_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load recovery patch module: {PATCH_PATH}")
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)


class RecoveryPatchTest(unittest.TestCase):
    def test_patch_is_valid_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiment"
            model_dir = experiment / "Code/src/model"
            recovery = root / "recovery"
            model_dir.mkdir(parents=True)
            recovery.mkdir()

            sharp_source = (
                "class Fake:\n"
                "    def forward(self, target_pos, data, x_encoder_all, target_pos_embed):\n"
                + patch.OLD
                + "            return target_encoder\n"
            )
            lightning_source = (
                "class Fake:\n"
                "    def cal_loss(self):\n"
                "        if True:\n"
                + patch.LOSS_OLD
                + "            'loss': loss,\n"
                "        }\n"
                "        return loss, disp_dict\n"
            )
            sharp_path = model_dir / "sharp.py"
            lightning_path = model_dir / "pl_modules.py"
            runner = experiment / "run_10_test_suite.sh"
            sharp_path.write_text(sharp_source, encoding="utf-8")
            lightning_path.write_text(lightning_source, encoding="utf-8")
            runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            script = Path(patch.__file__).resolve()
            command = [sys.executable, str(script), str(experiment), str(recovery)]
            subprocess.run(command, check=True)
            subprocess.run(command, check=True)

            patched_sharp = sharp_path.read_text(encoding="utf-8")
            patched_lightning = lightning_path.read_text(encoding="utf-8")
            compile(patched_sharp, str(sharp_path), "exec")
            compile(patched_lightning, str(lightning_path), "exec")
            self.assertEqual(patched_sharp.count(patch.MARKER), 1)
            self.assertEqual(patched_lightning.count(patch.LOSS_MARKER), 1)

            sharp_backup = recovery / "sharp.py.before_uncertainty_ddp_bridge"
            lightning_backup = recovery / "pl_modules.py.before_uncertainty_loss_bridge"
            self.assertEqual(sharp_backup.read_text(encoding="utf-8"), sharp_source)
            self.assertEqual(
                lightning_backup.read_text(encoding="utf-8"), lightning_source
            )


if __name__ == "__main__":
    unittest.main()
