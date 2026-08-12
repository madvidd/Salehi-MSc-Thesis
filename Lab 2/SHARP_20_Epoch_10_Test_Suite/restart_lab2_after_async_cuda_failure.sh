#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server00/M
PACKAGE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT_POINTER="$BASE/Codes/LATEST_SHARP_AV2_20EPOCH_10TEST_V4.txt"
RESULTS_POINTER="$BASE/Results/LATEST_SHARP_AV2_20EPOCH_10TEST.txt"

EXPERIMENT=$(tr -d '\r\n' < "$EXPERIMENT_POINTER" 2>/dev/null)
RESULTS=$(tr -d '\r\n' < "$RESULTS_POINTER" 2>/dev/null)

if [[ ! -d "$EXPERIMENT" || ! -d "$RESULTS" ]]; then
  echo "FATAL: Lab 2 experiment or results pointer is invalid."
  exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
RECOVERY="$RESULTS/recovery/async_cuda_failure_$STAMP"
mkdir -p "$RECOVERY"

[[ -f "$RESULTS/suite.log" ]] && cp --reflink=auto \
  "$RESULTS/suite.log" "$RECOVERY/suite.log"
[[ -f "$RESULTS/05_uncertainty_target_context/full_run.log" ]] && \
  cp --reflink=auto \
  "$RESULTS/05_uncertainty_target_context/full_run.log" \
  "$RECOVERY/test5_full_run.log"
find "$RESULTS" -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' | sort \
  > "$RECOVERY/CHECKPOINTS.txt"

suite_pids() {
  python3 - "$EXPERIMENT" "$RESULTS" <<'PY'
import pathlib
import sys

needles = tuple(sys.argv[1:])
for process in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        command = process.joinpath("cmdline").read_bytes()
        command = command.replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        continue
    if any(needle in command for needle in needles) and (
        "train.py" in command or "run_10_test_suite.sh" in command
    ):
        print(process.name)
PY
}

stop_suite_processes() {
  local signal=$1
  local wait_seconds=$2
  local pids
  pids=$(suite_pids)
  [[ -z "$pids" ]] && return 0
  echo "Sending SIG$signal to Lab 2 suite processes:"
  echo "$pids"
  kill -"$signal" $pids 2>/dev/null || true
  for ((second = 0; second < wait_seconds; second++)); do
    [[ -z "$(suite_pids)" ]] && return 0
    sleep 1
  done
}

stop_suite_processes INT 20
stop_suite_processes TERM 10
stop_suite_processes KILL 5

REMAINING=$(suite_pids)
if [[ -n "$REMAINING" ]]; then
  echo "FATAL: stale suite processes remain:"
  echo "$REMAINING"
  echo "Recovery evidence: $RECOVERY"
  exit 1
fi

echo "SUITE_PROCESSES_CLEARED=True"
echo "Failed attempt preserved at: $RECOVERY"
echo "Starting the validated foreground recovery..."

bash "$PACKAGE_DIR/recover_and_resume_lab2_10test_cuda.sh"
STATUS=$?

echo
echo "Lab 2 recovery status: $STATUS"
echo "Tests 1-4 and their checkpoints were not deleted."
echo "Terminal remains open."
exit "$STATUS"
