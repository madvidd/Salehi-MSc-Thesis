#!/usr/bin/env python3
"""Print progress for the resumable ten-variant suite."""

from __future__ import annotations

import re
import sys
from pathlib import Path


VARIANTS = (
    "01_baseline",
    "02_confidence_gated_memory",
    "03_cross_window_consistency",
    "04_learned_temporal_pool",
    "05_uncertainty_target_context",
    "06_relative_geometry_bias",
    "07_kinematic_motion_stem",
    "08_endpoint_refinement_decoder",
    "09_lane_topology_graph",
    "10_agent_temporal_mamba",
)


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch[_=](\d+)", path.name)
    return int(match.group(1)) if match else -1


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: suite_status.py RESULTS_ROOT")
    root = Path(sys.argv[1])
    print(f"Results root: {root}")
    print(f"{'Variant':38s} {'Status':12s} {'Last epoch':>10s} {'Checkpoints':>11s}")
    for variant in VARIANTS:
        directory = root / variant
        checkpoints = list(directory.rglob("*.ckpt")) if directory.exists() else []
        epochs = [checkpoint_epoch(path) for path in checkpoints]
        last_epoch = max(epochs, default=-1)
        if (directory / "COMPLETED").is_file():
            status = "COMPLETE"
        elif checkpoints:
            status = "RESUMABLE"
        elif directory.exists():
            status = "STARTED"
        else:
            status = "PENDING"
        print(f"{variant:38s} {status:12s} {last_epoch:10d} {len(checkpoints):11d}")


if __name__ == "__main__":
    main()
