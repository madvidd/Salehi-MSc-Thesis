#!/usr/bin/env bash
set -uo pipefail

BASE=/home/server00/M
PACKAGE="$BASE/Codes/Thesis/Lab 2/SHARP_Final_3_Run_Suite"
POINTER="$BASE/Codes/LATEST_SHARP_FINAL_3RUN_CODE.txt"
PYTHON_BIN="$BASE/Codes/envs/sharp/bin/python"
PREPARE="$PACKAGE/prepare_lab2_nvidia_58015903_userspace.sh"
STATUS=1

LOADED_NVIDIA=$(cat /sys/module/nvidia/version 2>/dev/null || true)
if [ "$LOADED_NVIDIA" = "580.159.03" ]; then
  bash "$PREPARE"
  STATUS=$?
  if [ "$STATUS" -eq 0 ]; then
    export LD_LIBRARY_PATH="$BASE/Codes/NVIDIA_USERSPACE_580.159.03/runtime/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$BASE/Codes/NVIDIA_USERSPACE_580.159.03/runtime/bin:$PATH"
  fi
else
  nvidia-smi -L
  STATUS=$?
fi

if [ "$STATUS" -eq 0 ]; then
  cd "$PACKAGE" || STATUS=1
fi
if [ "$STATUS" -eq 0 ]; then
  "$PYTHON_BIN" setup_final_3run_suite.py --base "$BASE"
  STATUS=$?
fi
if [ "$STATUS" -eq 0 ]; then
  ROOT=$(tr -d '\r\n' < "$POINTER")
  if [ ! -x "$ROOT/run_final_3run_suite.sh" ]; then
    echo "ERROR: generated final-suite runner is missing: $ROOT"
    STATUS=1
  fi
fi
if [ "$STATUS" -eq 0 ]; then
  cd "$ROOT" || STATUS=1
fi
if [ "$STATUS" -eq 0 ]; then
  bash "$ROOT/run_final_3run_suite.sh"
  STATUS=$?
fi

echo
echo "Final three-run suite status: $STATUS"
echo "The terminal remains open."
exit "$STATUS"
