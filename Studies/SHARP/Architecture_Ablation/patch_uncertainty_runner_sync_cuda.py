#!/usr/bin/env python3
"""Run only the uncertainty-context ablation with synchronous CUDA."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


MARKER = "CUDA_SYNCHRONOUS_EXECUTION=True"
ORIGINAL = '''  set +e
  "${COMMAND[@]}" 2>&1 | tee -a "$LOG"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e
'''
REPLACEMENT = '''  set +e
  if [[ "$VARIANT" == "uncertainty_target_context" ]]; then
    echo "CUDA_SYNCHRONOUS_EXECUTION=True variant=$VARIANT"
    CUDA_LAUNCH_BLOCKING=1 "${COMMAND[@]}" 2>&1 | tee -a "$LOG"
    TRAIN_STATUS=${PIPESTATUS[0]}
  else
    "${COMMAND[@]}" 2>&1 | tee -a "$LOG"
    TRAIN_STATUS=${PIPESTATUS[0]}
  fi
  set -e
'''


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: patch_uncertainty_runner_sync_cuda.py RUNNER RECOVERY_DIR")

    runner = Path(sys.argv[1]).resolve()
    recovery = Path(sys.argv[2]).resolve()
    recovery.mkdir(parents=True, exist_ok=True)

    source = runner.read_text()
    backup = recovery / "run_10_test_suite.sh.before_sync_cuda"
    if not backup.exists():
        shutil.copy2(runner, backup)

    if MARKER in source:
        print(f"UNCERTAINTY_RUNNER_SYNC_CUDA_ALREADY_ACTIVE={runner}")
        return

    count = source.count(ORIGINAL)
    if count != 1:
        raise RuntimeError(
            f"Expected one suite execution anchor in {runner}, found {count}"
        )

    runner.write_text(source.replace(ORIGINAL, REPLACEMENT, 1))
    updated = runner.read_text()
    if updated.count(MARKER) != 1 or "CUDA_LAUNCH_BLOCKING=1" not in updated:
        raise RuntimeError("Synchronous CUDA runner patch did not validate")

    print(f"UNCERTAINTY_RUNNER_SYNC_CUDA_PATCHED={runner}")
    print("SYNC_CUDA_VARIANT=uncertainty_target_context")
    print("OTHER_VARIANTS_ASYNC_CUDA=True")
    print("TRAINING_HYPERPARAMETERS_CHANGED=False")


if __name__ == "__main__":
    main()
