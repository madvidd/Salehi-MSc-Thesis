#!/usr/bin/env bash

# Resume only the combined QKNorm + Talking-Heads variant after an OOM. This
# script never sends signals to existing processes and refuses to launch a
# duplicate trainer or suite.

set -uo pipefail

BASE=/home/server01/M
ROOT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")
RESULTS_ROOT=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHECKPOINT="$RESULTS_ROOT/qknorm_talking_heads/run/checkpoints/last.ckpt"

echo "Lab 3 OOM-safe qknorm_talking_heads recovery"
echo "Results: $RESULTS_ROOT"
echo "Checkpoint: $CHECKPOINT"

ACTIVE_VARIANT=$(pgrep -af \
  '[t]rain[.]py.*qknorm_talking_heads|[r]un_variant[.]sh qknorm_talking_heads' \
  || true)
ACTIVE_SUITE=$(pgrep -af '[r]un_remaining_attention_suite' || true)

if [[ -n "$ACTIVE_VARIANT" || -n "$ACTIVE_SUITE" ]]; then
  echo "RECOVERY_NOT_STARTED: an existing trainer or suite process is still active."
  [[ -n "$ACTIVE_VARIANT" ]] && printf '%s\n' "$ACTIVE_VARIANT"
  [[ -n "$ACTIVE_SUITE" ]] && printf '%s\n' "$ACTIVE_SUITE"
  echo "No process was terminated and no duplicate trainer was started."
  exit 3
fi

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "FATAL: resumable checkpoint not found: $CHECKPOINT"
  exit 1
fi

ENV=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ENV.txt")
"$ENV/bin/python" - "$CHECKPOINT" <<'PY'
import sys

import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if "state_dict" not in checkpoint or "epoch" not in checkpoint:
    raise SystemExit("FATAL: checkpoint is incomplete")
print(f"CHECKPOINT_VALID={path}")
print(f"CHECKPOINT_EPOCH={checkpoint['epoch']}")
print(f"CHECKPOINT_GLOBAL_STEP={checkpoint.get('global_step')}")
PY

CHECK_STATUS=$?
if (( CHECK_STATUS != 0 )); then
  echo "FATAL: checkpoint validation failed with status $CHECK_STATUS"
  exit "$CHECK_STATUS"
fi

echo "First retry: original per-GPU batch 8 and global batch 24."
echo "Only the expandable CUDA allocator changes for this attempt."
echo "The existing results and checkpoints remain in place."

BATCH_PER_GPU=8 \
ACCUMULATE_GRAD_BATCHES=1 \
LAB3_VARIANTS=qknorm_talking_heads \
LAB3_MAX_ATTEMPTS=1 \
bash "$SCRIPT_DIR/run_remaining_attention_suite.sh"

STATUS=$?

if (( STATUS != 0 )); then
  LATEST_ATTEMPT=$(ls -t \
    "$RESULTS_ROOT/qknorm_talking_heads/attempt_logs"/attempt_1_*.log \
    2>/dev/null | head -1)

  if [[ -z "$LATEST_ATTEMPT" ]] || \
     ! grep -aEq 'CUDA out of memory|OutOfMemoryError' "$LATEST_ATTEMPT"; then
    echo "The exact-configuration retry failed for a non-OOM reason."
    echo "Automatic fallback was not started."
  else
    echo "The exact-configuration retry encountered another CUDA OOM."

    ACTIVE_VARIANT=$(pgrep -af \
      '[t]rain[.]py.*qknorm_talking_heads|[r]un_variant[.]sh qknorm_talking_heads' \
      || true)
    ACTIVE_SUITE=$(pgrep -af '[r]un_remaining_attention_suite' || true)

    if [[ -n "$ACTIVE_VARIANT" || -n "$ACTIVE_SUITE" ]]; then
      echo "FALLBACK_NOT_STARTED: failed processes have not exited."
      [[ -n "$ACTIVE_VARIANT" ]] && printf '%s\n' "$ACTIVE_VARIANT"
      [[ -n "$ACTIVE_SUITE" ]] && printf '%s\n' "$ACTIVE_SUITE"
      echo "No process was terminated."
    else
      echo "Fallback: per-GPU batch 4 with accumulation 2."
      echo "Effective optimization batch remains 4 x 3 x 2 = 24."
      echo "SyncBatchNorm microbatch statistics will differ from batch 8."

      BATCH_PER_GPU=4 \
      ACCUMULATE_GRAD_BATCHES=2 \
      LAB3_VARIANTS=qknorm_talking_heads \
      LAB3_MAX_ATTEMPTS=4 \
      bash "$SCRIPT_DIR/run_remaining_attention_suite.sh"

      STATUS=$?
    fi
  fi
fi

echo
echo "OOM-safe resume status: $STATUS"
echo "Terminal remains open."
exit "$STATUS"
