#!/usr/bin/env python3
"""Regression tests for the checked uncertainty target-context remap patch."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import patch_uncertainty_target_context_cuda as patcher


class CheckedTargetRemapPatchTests(unittest.TestCase):
    def test_patch_is_exact_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            recovery = Path(temporary) / "recovery"
            model = root / "Code/src/model/sharp.py"
            model.parent.mkdir(parents=True)
            model.write_text(
                "import torch\n\n"
                "def remap(container, target_valid, compressed_target_encoder, "
                "compressed_target_mask, self):\n"
                + patcher.OLD,
                encoding="utf-8",
            )

            command = [sys.executable, patcher.__file__, str(root), str(recovery)]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            second = subprocess.run(command, check=True, capture_output=True, text=True)

            patched = model.read_text(encoding="utf-8")
            self.assertIn("UNCERTAINTY_TARGET_REMAP=APPLIED", first.stdout)
            self.assertIn("UNCERTAINTY_TARGET_REMAP=ALREADY_APPLIED", second.stdout)
            self.assertEqual(patched.count(patcher.MARKER), 1)
            self.assertNotIn(patcher.OLD, patched)
            compile(patched, str(model), "exec")

            backup = recovery / "sharp.py.before_checked_target_remap"
            self.assertEqual(
                backup.read_text(encoding="utf-8"),
                "import torch\n\n"
                "def remap(container, target_valid, compressed_target_encoder, "
                "compressed_target_mask, self):\n"
                + patcher.OLD,
            )

    def test_unknown_source_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            recovery = Path(temporary) / "recovery"
            model = root / "Code/src/model/sharp.py"
            model.parent.mkdir(parents=True)
            original = "value = 1\n"
            model.write_text(original, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, patcher.__file__, str(root), str(recovery)],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(model.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
