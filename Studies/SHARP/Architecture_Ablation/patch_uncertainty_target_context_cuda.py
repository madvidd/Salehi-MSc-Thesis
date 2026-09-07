#!/usr/bin/env python3
"""Replace the failing target-context Boolean CUDA write with checked index_copy."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OLD = """            container[target_valid.view(-1)] = compressed_target_encoder.view(-1, self.embed_dim)[~compressed_target_mask.view(-1)]
"""

NEW = """            # Map the padded target features back without a Boolean CUDA write.
            # Both index tensors are row-major, so this is equivalent to the
            # original assignment while avoiding its failing CUDA kernel path.
            target_destination = torch.nonzero(
                target_valid.reshape(-1), as_tuple=False
            ).squeeze(-1)
            compressed_source = torch.nonzero(
                ~compressed_target_mask.reshape(-1), as_tuple=False
            ).squeeze(-1)
            if target_destination.numel() != compressed_source.numel():
                raise RuntimeError(
                    "Target-context remap count mismatch: "
                    f"destination={target_destination.numel()} "
                    f"source={compressed_source.numel()}"
                )
            compressed_flat = compressed_target_encoder.reshape(
                -1, self.embed_dim
            ).index_select(0, compressed_source)
            container = torch.index_copy(
                container, 0, target_destination, compressed_flat
            )
"""

MARKER = "target_destination = torch.nonzero("


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("recovery_directory", type=Path)
    args = parser.parse_args()

    experiment = args.experiment_root.resolve()
    recovery = args.recovery_directory.resolve()
    model_path = experiment / "Code/src/model/sharp.py"

    if not model_path.is_file():
        raise SystemExit(f"FATAL: SHARP model file is missing: {model_path}")

    recovery.mkdir(parents=True, exist_ok=True)
    backup = recovery / "sharp.py.before_checked_target_remap"
    if not backup.exists():
        shutil.copy2(model_path, backup)

    source = model_path.read_text(encoding="utf-8")
    if MARKER in source:
        status = "ALREADY_APPLIED"
    else:
        count = source.count(OLD)
        if count != 1:
            raise SystemExit(
                "FATAL: expected one target-context remap anchor, "
                f"found {count}; the model was not changed"
            )
        source = source.replace(OLD, NEW)
        compile(source, str(model_path), "exec")
        model_path.write_text(source, encoding="utf-8", newline="\n")
        status = "APPLIED"

    verified = model_path.read_text(encoding="utf-8")
    if verified.count(MARKER) != 1 or OLD in verified:
        raise SystemExit("FATAL: checked target remap verification failed")
    compile(verified, str(model_path), "exec")

    (recovery / "TARGET_REMAP_PATCH_STATUS.txt").write_text(
        "\n".join(
            (
                f"status={status}",
                f"experiment={experiment}",
                f"model={model_path}",
                f"backup={backup}",
                "mathematical_operation=unchanged",
                "training_configuration=unchanged",
                "completed_results_changed=none",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"UNCERTAINTY_TARGET_REMAP={status}")
    print(f"PATCHED_MODEL={model_path}")
    print(f"MODEL_BACKUP={backup}")


if __name__ == "__main__":
    main()
