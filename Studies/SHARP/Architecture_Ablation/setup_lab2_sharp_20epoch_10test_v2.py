#!/usr/bin/env python3
"""Compatibility entry point for Lab 2 SHARP checkpoint configurations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
ORIGINAL_SETUP = HERE / "setup_lab2_sharp_20epoch_10test.py"

spec = importlib.util.spec_from_file_location("sharp_10test_setup", ORIGINAL_SETUP)
if spec is None or spec.loader is None:
    raise SystemExit(f"Could not load setup module: {ORIGINAL_SETUP}")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

original_patch_common = setup.patch_common


def patch_common_compatible(code_dir: Path) -> None:
    """Normalize any integer save_top_k value before applying strict patches."""
    config_path = code_dir / "conf/config.yaml"
    config = config_path.read_text(encoding="utf-8")
    config, count = re.subn(
        r"(?m)^    save_top_k:\s*-?\d+\s*$",
        "    save_top_k: 10",
        config,
    )
    if count != 1:
        raise RuntimeError(
            "Expected one integer save_top_k setting in copied SHARP config, "
            f"found {count}"
        )
    config_path.write_text(config, encoding="utf-8", newline="\n")
    original_patch_common(code_dir)


setup.patch_common = patch_common_compatible
setup.main()
