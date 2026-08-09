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

echo "Resuming with per-GPU batch 4 and accumulation 2."
echo "Effective global batch remains 4 x 3 x 2 = 24."
echo "The existing results and checkpoints remain in place."

BATCH_PER_GPU=4 \
ACCUMULATE_GRAD_BATCHES=2 \
LAB3_VARIANTS=qknorm_talking_heads \
LAB3_MAX_ATTEMPTS=4 \
bash "$SCRIPT_DIR/run_remaining_attention_suite.sh"

STATUS=$?
echo
echo "OOM-safe resume status: $STATUS"
echo "Terminal remains open."
exit "$STATUS"
