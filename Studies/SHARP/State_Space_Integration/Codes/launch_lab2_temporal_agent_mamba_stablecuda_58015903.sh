#!/usr/bin/env bash
set -euo pipefail

BASE=/home/server00/M
REPO="$BASE/Codes/Thesis"
SETUP="$REPO/Studies/SHARP/State_Space_Integration/Codes/setup_lab2_temporal_agent_mamba_stablecuda80.py"
PREPARE="$REPO/Studies/SHARP/State_Space_Integration/Codes/prepare_lab2_nvidia_58015903_userspace.sh"
COMPAT_ROOT="$BASE/Codes/NVIDIA_USERSPACE_580.159.03"
COMPAT_LIB="$COMPAT_ROOT/runtime/lib"
COMPAT_BIN="$COMPAT_ROOT/runtime/bin"
PATTERN='SHARP_AV2_(TEMPORAL_AGENT_MAMBA(8[0]|_CHUNKED8[0]|_STABLECUDA8[0])|MAMBA_FUSED8[0])|run_(temporal_agent_mamba(_chunked|_stablecuda)?_4gpu|av2_mamba_fused_4gpu_syncbn)[.]sh'

test -f "$SETUP"
test -f "$PREPARE"
test -d "$BASE/Codes/SHARP/Code"
test -x "$BASE/Codes/envs/sharp/bin/python"
test -d "$BASE/Datasets/AV2/sharp_processed/train"
test -d "$BASE/Datasets/AV2/sharp_processed/val"

PIDS=$(pgrep -u "$USER" -f "$PATTERN" || true)
if [ -n "$PIDS" ]; then
  echo "Stopping earlier Lab 2 SHARP Mamba jobs: $PIDS"
  kill -INT $PIDS 2>/dev/null || true
  sleep 10
  PIDS=$(pgrep -u "$USER" -f "$PATTERN" || true)
fi
if [ -n "$PIDS" ]; then
  echo "Earlier jobs are still active; sending TERM: $PIDS"
  kill -TERM $PIDS 2>/dev/null || true
  sleep 10
  PIDS=$(pgrep -u "$USER" -f "$PATTERN" || true)
fi
if [ -n "$PIDS" ]; then
  echo "ERROR: earlier SHARP Mamba jobs remain active: $PIDS"
  exit 1
fi

bash "$PREPARE"

test -s "$COMPAT_ROOT/READY.txt"
export LD_LIBRARY_PATH="$COMPAT_LIB:${LD_LIBRARY_PATH:-}"
export PATH="$COMPAT_BIN:$PATH"

echo "Validated GPU visibility:"
nvidia-smi -L

cd "$REPO/Studies/SHARP/State_Space_Integration/Codes"
"$BASE/Codes/envs/sharp/bin/python" \
  setup_lab2_temporal_agent_mamba_stablecuda80.py

ROOT=$(cat "$BASE/Codes/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_STABLECUDA80.txt")
RESULTS=$(cat "$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_STABLECUDA80.txt")
RUNNER="$ROOT/run_temporal_agent_mamba_stablecuda_4gpu.sh"

echo "New code:    $ROOT"
echo "New results: $RESULTS"
cat "$ROOT/EXPERIMENT.txt"

cd "$ROOT/Code"
echo "Starting stable-CUDA temporal-agent Mamba training in this foreground terminal."
bash "$RUNNER"
STATUS=$?

echo
echo "Training finished or stopped with status: $STATUS"
exit "$STATUS"
