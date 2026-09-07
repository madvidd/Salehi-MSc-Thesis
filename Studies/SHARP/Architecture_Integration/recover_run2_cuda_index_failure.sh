#!/usr/bin/env bash
set -uo pipefail

BASE=/home/server00/M
PACKAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_POINTER="$BASE/Codes/LATEST_SHARP_FINAL_3RUN_CODE.txt"
RESULTS_POINTER="$BASE/Results/LATEST_SHARP_FINAL_3RUN_RESULTS.txt"
PYTHON_BIN="$BASE/Codes/envs/sharp/bin/python"
STATUS=1

EXPERIMENT_ROOT=$(tr -d '\r\n' < "$CODE_POINTER" 2>/dev/null || true)
RESULTS_ROOT=$(tr -d '\r\n' < "$RESULTS_POINTER" 2>/dev/null || true)
RUN1="$RESULTS_ROOT/01_official_sharp_baseline"
RUN2="$RESULTS_ROOT/02_qknorm_uncertainty_geometry"

if [ ! -d "$EXPERIMENT_ROOT" ] || [ ! -d "$RESULTS_ROOT" ]; then
  echo "ERROR: existing final-suite pointers are invalid."
  exit 1
fi
if [ ! -s "$RUN1/COMPLETED" ] || [ ! -s "$RUN1/artifacts/metrics.json" ]; then
  echo "ERROR: completed Run 1 evidence is missing; recovery was not attempted."
  exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
RECOVERY="$RESULTS_ROOT/runtime_repairs/run2_illegal_cuda_$STAMP"
mkdir -p "$RECOVERY"
cp --reflink=auto -p "$RESULTS_ROOT/suite.log" "$RECOVERY/suite.log" 2>/dev/null || true
cp --reflink=auto -p "$RUN2/full_run.log" "$RECOVERY/run2_full_run.log" 2>/dev/null || true
cp -a "$RUN2/attempt_logs" "$RECOVERY/" 2>/dev/null || true
find "$RESULTS_ROOT" -type f -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' | sort \
  > "$RECOVERY/CHECKPOINTS.txt"
ps -u "$USER" -o pid,ppid,pgid,etime,state,%cpu,%mem,args \
  > "$RECOVERY/PROCESSES_BEFORE.txt"

find_suite_pids() {
  "$PYTHON_BIN" - "$EXPERIMENT_ROOT" "$RESULTS_ROOT" <<'PY'
import os
import pathlib
import sys

needles = tuple(sys.argv[1:])
for proc in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        pid = int(proc.name)
        if pid in (os.getpid(), os.getppid()):
            continue
        command = proc.joinpath("cmdline").read_bytes()
        command = command.replace(b"\0", b" ").decode(errors="ignore")
    except (OSError, ValueError):
        continue
    if not any(needle in command for needle in needles):
        continue
    if "train.py" in command or "run_final_3run_suite.sh" in command:
        print(pid)
PY
}

PIDS=$(find_suite_pids)
if [ -n "$PIDS" ]; then
  echo "Stopping only the failed final-suite Run 2 processes:"
  echo "$PIDS"
  kill -INT $PIDS 2>/dev/null || true
  for _ in $(seq 1 20); do
    sleep 2
    [ -z "$(find_suite_pids)" ] && break
  done
fi

PIDS=$(find_suite_pids)
if [ -n "$PIDS" ]; then
  echo "Terminating remaining failed ranks:"
  echo "$PIDS"
  kill -TERM $PIDS 2>/dev/null || true
  for _ in $(seq 1 15); do
    sleep 2
    [ -z "$(find_suite_pids)" ] && break
  done
fi

PIDS=$(find_suite_pids)
if [ -n "$PIDS" ]; then
  echo "Force-clearing unresponsive failed ranks:"
  echo "$PIDS"
  kill -KILL $PIDS 2>/dev/null || true
  sleep 5
fi

PIDS=$(find_suite_pids)
if [ -n "$PIDS" ]; then
  echo "ERROR: failed Run 2 processes still remain:"
  echo "$PIDS"
  exit 1
fi

sync
echo "FAILED_RUN2_EVIDENCE_PRESERVED=$RECOVERY"
echo "SUITE_PROCESSES_CLEARED=YES"
echo "RUN1_PRESERVED=YES"
echo "Run 2 has no completed checkpoint and will restart from its beginning."
echo "Run 1 will not be retrained. Run 3 will use the same CUDA-safe indexing."

cd "$PACKAGE" || exit 1
bash "$PACKAGE/launch_final_3run_suite.sh"
STATUS=$?

echo
echo "Run 2 recovery status: $STATUS"
echo "Recovery evidence: $RECOVERY"
echo "The terminal remains open."
exit "$STATUS"
