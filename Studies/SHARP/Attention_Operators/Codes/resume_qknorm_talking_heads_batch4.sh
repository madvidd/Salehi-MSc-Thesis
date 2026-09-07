#!/usr/bin/env bash

# Clear only stale combined-attention processes and resume its saved checkpoint
# with a smaller microbatch while preserving the effective optimization batch.

set -uo pipefail

BASE=/home/server01/M
RESULTS_ROOT=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT="$RESULTS_ROOT/qknorm_talking_heads"
CHECKPOINT="$OUT/run/checkpoints/last.ckpt"
ENV=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ENV.txt")
STAMP=$(date +%Y%m%d-%H%M%S)
PRESERVE="$RESULTS_ROOT/recovery_before_batch4_$STAMP"

collect_stale_pids() {
  ps -u "$USER" -o pid=,args= | awk -v out="$OUT" '
    index($0, out) && ($0 ~ /[t]rain[.]py/ || $0 ~ /[t]ee -a/) {print $1}
    /[r]un_variant[.]sh qknorm_talking_heads/ {print $1}
    /[r]un_remaining_attention_suite(_resilient)?[.]sh/ {print $1}
  ' | sort -un
}

if [[ ! -s "$CHECKPOINT" ]]; then
  echo "FATAL: checkpoint not found: $CHECKPOINT"
  exit 1
fi

"$ENV/bin/python" - "$CHECKPOINT" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
if "state_dict" not in checkpoint or "epoch" not in checkpoint:
    raise SystemExit("FATAL: incomplete checkpoint")
print(f"CHECKPOINT_VALID={path}")
print(f"CHECKPOINT_EPOCH={checkpoint['epoch']}")
print(f"CHECKPOINT_GLOBAL_STEP={checkpoint.get('global_step')}")
PY

if (( $? != 0 )); then
  echo "FATAL: checkpoint validation failed. No process was changed."
  exit 1
fi

mkdir -p "$PRESERVE"
cp -p "$CHECKPOINT" "$PRESERVE/last.ckpt"
find "$OUT/attempt_logs" -maxdepth 1 -type f -name 'attempt_*.log' \
  -exec cp -p {} "$PRESERVE/" \; 2>/dev/null || true
printf 'preserved=%s\ncheckpoint=%s\n' \
  "$(date --iso-8601=seconds)" "$CHECKPOINT" > "$PRESERVE/RECOVERY.txt"

PIDS=$(collect_stale_pids)
if [[ -n "$PIDS" ]]; then
  echo "Stopping only stale qknorm_talking_heads suite processes:"
  printf '%s\n' "$PIDS"
  kill -TERM $PIDS 2>/dev/null || true
  for _ in $(seq 1 20); do
    [[ -z "$(collect_stale_pids)" ]] && break
    sleep 2
  done
fi

PIDS=$(collect_stale_pids)
if [[ -n "$PIDS" ]]; then
  echo "Force-clearing unresponsive stale ranks:"
  printf '%s\n' "$PIDS"
  kill -KILL $PIDS 2>/dev/null || true
  sleep 10
fi

PIDS=$(collect_stale_pids)
if [[ -n "$PIDS" ]]; then
  echo "FATAL: stale processes remain:"
  printf '%s\n' "$PIDS"
  exit 1
fi

echo "PRESERVED_FAILURE_STATE=$PRESERVE"
echo "Resuming with per-GPU batch 4 and gradient accumulation 2."
echo "Effective optimization batch: 4 x 3 x 2 = 24."
export TORCH_SHOW_CPP_STACKTRACES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

BATCH_PER_GPU=4 \
ACCUMULATE_GRAD_BATCHES=2 \
LAB3_VARIANTS=qknorm_talking_heads \
LAB3_MAX_ATTEMPTS=4 \
bash "$SCRIPT_DIR/run_remaining_attention_suite.sh"
STATUS=$?

echo
echo "Batch-4 recovery status: $STATUS"
echo "Terminal remains open."
exit "$STATUS"
