#!/usr/bin/env bash
set -uo pipefail

BASE=/home/server00/M
PACKAGE="$BASE/Codes/Thesis/Lab 2/SHARP_20_Epoch_10_Test_Suite"
POINTER="$BASE/Codes/LATEST_SHARP_AV2_20EPOCH_10TEST.txt"
PREPARE="$PACKAGE/prepare_lab2_nvidia_58015903_userspace.sh"
PYTHON_BIN="$BASE/Codes/envs/sharp/bin/python"
STATUS=1

LOADED_NVIDIA=$(cat /sys/module/nvidia/version 2>/dev/null || true)
if [ "$LOADED_NVIDIA" = "580.159.03" ]; then
  bash "$PREPARE"
  STATUS=$?
  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: isolated NVIDIA 580.159.03 runtime preparation failed."
  else
    export LD_LIBRARY_PATH="$BASE/Codes/NVIDIA_USERSPACE_580.159.03/runtime/lib:${LD_LIBRARY_PATH:-}"
    export PATH="$BASE/Codes/NVIDIA_USERSPACE_580.159.03/runtime/bin:$PATH"
  fi
else
  nvidia-smi -L
  STATUS=$?
fi

if [ "$STATUS" -eq 0 ]; then
  ROOT=""
  if [ -s "$POINTER" ]; then
    CANDIDATE=$(tr -d '\r\n' < "$POINTER")
    if [ -x "$CANDIDATE/run_10_test_suite.sh" ]; then
      ROOT="$CANDIDATE"
      echo "Reusing resumable experiment: $ROOT"
    fi
  fi

  if [ -z "$ROOT" ]; then
    cd "$PACKAGE" || STATUS=1
    if [ "$STATUS" -eq 0 ]; then
      "$PYTHON_BIN" setup_lab2_sharp_20epoch_10test.py
      STATUS=$?
    fi
    if [ "$STATUS" -eq 0 ]; then
      ROOT=$(tr -d '\r\n' < "$POINTER")
    fi
  fi

  if [ "$STATUS" -eq 0 ]; then
    cd "$ROOT/Code" || STATUS=1
  fi
  if [ "$STATUS" -eq 0 ]; then
    bash "$ROOT/run_10_test_suite.sh"
    STATUS=$?
  fi
fi

echo
echo "Foreground suite returned status: $STATUS"
echo "Terminal remains open."
exit "$STATUS"
