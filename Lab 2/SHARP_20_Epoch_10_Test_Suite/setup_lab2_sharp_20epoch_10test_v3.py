#!/usr/bin/env python3
"""Create the ten-test suite with Lab 2 config and optimizer-name compatibility."""

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


def replace_exactly_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor in {path}, found {count}")
    path.write_text(source.replace(old, new), encoding="utf-8", newline="\n")


setup.patch_common = patch_common_compatible
setup.main()

base = Path("/home/server00/M")
pointer = base / "Codes/LATEST_SHARP_AV2_20EPOCH_10TEST.txt"
experiment_root = Path(pointer.read_text(encoding="utf-8").strip())

# SHARP's optimizer groups inspect the complete parameter name with a substring
# check for "bias". Keeping that word in a module attribute makes its Linear
# weights appear in both decay and no-decay groups. Rename only the attribute;
# the RelativeGeometryBias module and the ablation behavior remain unchanged.
sharp_path = experiment_root / "Code/src/model/sharp.py"
sharp_source = sharp_path.read_text(encoding="utf-8")
attribute_count = sharp_source.count("self.relative_geometry_bias")
if attribute_count != 3:
    raise RuntimeError(
        "Expected three relative-geometry attribute references in generated "
        f"SHARP, found {attribute_count}"
    )
sharp_path.write_text(
    sharp_source.replace(
        "self.relative_geometry_bias", "self.relative_geometry_prior"
    ),
    encoding="utf-8",
    newline="\n",
)

replace_exactly_once(
    experiment_root / "Code/train.py",
    '"relative_geometry_bias": "relative_geometry_bias"',
    '"relative_geometry_bias": "relative_geometry_prior"',
    "runtime relative-geometry marker",
)
replace_exactly_once(
    experiment_root / "preflight_suite.py",
    '"relative_geometry_bias": "relative_geometry_bias"',
    '"relative_geometry_bias": "relative_geometry_prior"',
    "preflight relative-geometry marker",
)

v3_pointer = base / "Codes/LATEST_SHARP_AV2_20EPOCH_10TEST_V3.txt"
v3_pointer.write_text(str(experiment_root) + "\n", encoding="utf-8", newline="\n")

print("RELATIVE_GEOMETRY_OPTIMIZER_NAME_FIX_APPLIED")
print(f"V3_EXPERIMENT_ROOT={experiment_root}")
