#!/usr/bin/env python3
"""Patch the current Lab 2 experiment for conditional DDP parameters."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OLD = '''            if self.uncertainty_target_context is not None and "pi" in data["memory_dict"]:
                max_distance, target_feature_gate = self.uncertainty_target_context(
                    data["memory_dict"]["pi"].float()
                )
            target_encoder = x_encoder_all.unsqueeze(1).expand_as(target_pos_embed) + target_pos_embed
'''

NEW = '''            if self.uncertainty_target_context is not None and "pi" in data["memory_dict"]:
                max_distance, target_feature_gate = self.uncertainty_target_context(
                    data["memory_dict"]["pi"].float()
                )
            elif self.uncertainty_target_context is not None:
                # Keep conditionally inactive parameters in the DDP graph without
                # changing forward values before streamed probabilities exist.
                uncertainty_ddp_zero = target_pos.new_zeros(())
                for parameter in self.uncertainty_target_context.parameters():
                    uncertainty_ddp_zero = uncertainty_ddp_zero + parameter.sum() * 0.0
                target_feature_gate = target_feature_gate + uncertainty_ddp_zero
            target_encoder = x_encoder_all.unsqueeze(1).expand_as(target_pos_embed) + target_pos_embed
'''

MARKER = "uncertainty_ddp_zero = target_pos.new_zeros(())"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("recovery_directory", type=Path)
    args = parser.parse_args()

    experiment = args.experiment_root.resolve()
    recovery = args.recovery_directory.resolve()
    sharp_path = experiment / "Code/src/model/sharp.py"
    runner = experiment / "run_10_test_suite.sh"

    if not sharp_path.is_file():
        raise SystemExit(f"FATAL: SHARP model file is missing: {sharp_path}")
    if not runner.is_file():
        raise SystemExit(f"FATAL: suite runner is missing: {runner}")

    recovery.mkdir(parents=True, exist_ok=True)
    backup = recovery / "sharp.py.before_uncertainty_ddp_bridge"
    if not backup.exists():
        shutil.copy2(sharp_path, backup)
    runner_backup = recovery / "run_10_test_suite.sh"
    if not runner_backup.exists():
        shutil.copy2(runner, runner_backup)

    source = sharp_path.read_text(encoding="utf-8")
    if MARKER in source:
        status = "ALREADY_APPLIED"
    else:
        count = source.count(OLD)
        if count != 1:
            raise SystemExit(
                "FATAL: expected one uncertainty-context patch anchor, "
                f"found {count}; the model was not changed"
            )
        source = source.replace(OLD, NEW)
        compile(source, str(sharp_path), "exec")
        sharp_path.write_text(source, encoding="utf-8", newline="\n")
        status = "APPLIED"

    verified = sharp_path.read_text(encoding="utf-8")
    if verified.count(MARKER) != 1:
        raise SystemExit("FATAL: uncertainty DDP bridge verification failed")
    compile(verified, str(sharp_path), "exec")

    manifest = recovery / "PATCH_STATUS.txt"
    manifest.write_text(
        "\n".join(
            (
                f"status={status}",
                f"experiment={experiment}",
                f"model={sharp_path}",
                f"backup={backup}",
                "forward_value_change=none",
                "ddp_strategy_change=none",
                "completed_result_directories_changed=none",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"UNCERTAINTY_DDP_BRIDGE={status}")
    print(f"PATCHED_MODEL={sharp_path}")
    print(f"ORIGINAL_MODEL_BACKUP={backup}")


if __name__ == "__main__":
    main()
