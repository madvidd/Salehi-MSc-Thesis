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
LOSS_OLD = '''            loss = loss + 0.2 * coarse_loss
        disp_dict = {
'''
LOSS_NEW = '''            loss = loss + 0.2 * coarse_loss
        if self.experiment_variant == 'uncertainty_target_context':
            for parameter in self.model.uncertainty_target_context.parameters():
                loss = loss + parameter.sum() * 0.0
        disp_dict = {
'''
LOSS_MARKER = "loss = loss + parameter.sum() * 0.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("recovery_directory", type=Path)
    args = parser.parse_args()

    experiment = args.experiment_root.resolve()
    recovery = args.recovery_directory.resolve()
    sharp_path = experiment / "Code/src/model/sharp.py"
    lightning_path = experiment / "Code/src/model/pl_modules.py"
    runner = experiment / "run_10_test_suite.sh"

    if not sharp_path.is_file():
        raise SystemExit(f"FATAL: SHARP model file is missing: {sharp_path}")
    if not lightning_path.is_file():
        raise SystemExit(f"FATAL: Lightning module file is missing: {lightning_path}")
    if not runner.is_file():
        raise SystemExit(f"FATAL: suite runner is missing: {runner}")

    recovery.mkdir(parents=True, exist_ok=True)
    backup = recovery / "sharp.py.before_uncertainty_ddp_bridge"
    if not backup.exists():
        shutil.copy2(sharp_path, backup)
    runner_backup = recovery / "run_10_test_suite.sh"
    if not runner_backup.exists():
        shutil.copy2(runner, runner_backup)
    lightning_backup = recovery / "pl_modules.py.before_uncertainty_loss_bridge"
    if not lightning_backup.exists():
        shutil.copy2(lightning_path, lightning_backup)

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

    lightning = lightning_path.read_text(encoding="utf-8")
    if LOSS_MARKER in lightning:
        loss_status = "ALREADY_APPLIED"
    else:
        count = lightning.count(LOSS_OLD)
        if count != 1:
            raise SystemExit(
                "FATAL: expected one uncertainty loss-bridge anchor, "
                f"found {count}; the Lightning module was not changed"
            )
        lightning = lightning.replace(LOSS_OLD, LOSS_NEW)
        compile(lightning, str(lightning_path), "exec")
        lightning_path.write_text(lightning, encoding="utf-8", newline="\n")
        loss_status = "APPLIED"

    verified_lightning = lightning_path.read_text(encoding="utf-8")
    if verified_lightning.count(LOSS_MARKER) != 1:
        raise SystemExit("FATAL: uncertainty loss bridge verification failed")
    compile(verified_lightning, str(lightning_path), "exec")

    manifest = recovery / "PATCH_STATUS.txt"
    manifest.write_text(
        "\n".join(
            (
                f"status={status}",
                f"loss_bridge_status={loss_status}",
                f"experiment={experiment}",
                f"model={sharp_path}",
                f"lightning_module={lightning_path}",
                f"backup={backup}",
                f"lightning_backup={lightning_backup}",
                "forward_value_change=none",
                "ddp_strategy_change=none",
                "completed_result_directories_changed=none",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"UNCERTAINTY_DDP_BRIDGE={status}")
    print(f"UNCERTAINTY_LOSS_BRIDGE={loss_status}")
    print(f"PATCHED_MODEL={sharp_path}")
    print(f"PATCHED_LIGHTNING_MODULE={lightning_path}")
    print(f"ORIGINAL_MODEL_BACKUP={backup}")
    print(f"ORIGINAL_LIGHTNING_BACKUP={lightning_backup}")


if __name__ == "__main__":
    main()
