#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

BASE=${BASE:-/home/server01/M}
RESULTS=$(tr -d '\r\n' < "$BASE/Results/LATEST_SEAM_AV2_20EPOCH_4TEST.txt" 2>/dev/null)
EXPERIMENT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SEAM_AV2_20EPOCH_4TEST_CODE.txt" 2>/dev/null)
STAMP=$(date +%Y%m%d-%H%M%S)

if [ -z "$RESULTS" ] || [ ! -d "$RESULTS" ]; then
  echo "No earlier batch-32 SEAM result directory was found."
  echo "Terminal remains open."
  exit 0
fi

ARCHIVE="$BASE/Terminal/SEAM_20_Epoch_4_Test_Suite/$(basename "$RESULTS")/Before_Max_Resources_$STAMP"
mkdir -p "$ARCHIVE"

if [ -s "$RESULTS/suite.log" ]; then
  tr '\r' '\n' < "$RESULTS/suite.log" > "$ARCHIVE/Terminal.txt"
fi
cp -p "$RESULTS/Summary.md" "$RESULTS/RUN_STATUS.txt" "$ARCHIVE/" 2>/dev/null
find "$RESULTS" -type f -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' 2>/dev/null | \
  sort > "$ARCHIVE/CHECKPOINTS.txt"
find "$RESULTS" -type f -name 'events.out.tfevents*' \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS  %s bytes  %p\n' 2>/dev/null | \
  sort > "$ARCHIVE/TENSORBOARD_EVENTS.txt"
printf 'saved=%s\nresults=%s\nexperiment=%s\n' \
  "$(date --iso-8601=seconds)" "$RESULTS" "$EXPERIMENT" \
  > "$ARCHIVE/SNAPSHOT_INFO.txt"
sync

find_pids() {
  python3 - "$RESULTS" "$EXPERIMENT" <<'PY'
import os
from pathlib import Path
import sys

needles = [value.encode() for value in sys.argv[1:] if value]
excluded = {os.getpid(), os.getppid()}
for proc in Path("/proc").glob("[0-9]*"):
    try:
        pid = int(proc.name)
        command = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ")
    except (OSError, ValueError):
        continue
    if pid in excluded or not any(needle in command for needle in needles):
        continue
    if b"train.py" in command or b"run_seam_20epoch_4test_suite.sh" in command or b"tee -a" in command:
        print(pid)
PY
}

PIDS=$(find_pids)
if [ -n "$PIDS" ]; then
  echo "Preserved the superseded run at: $ARCHIVE"
  echo "Requesting a graceful stop from only these processes:"
  echo "$PIDS"
  kill -INT $PIDS 2>/dev/null
  for _ in $(seq 1 18); do
    sleep 5
    [ -z "$(find_pids)" ] && break
  done
fi

PIDS=$(find_pids)
if [ -n "$PIDS" ]; then
  echo "Graceful stop timed out; terminating the remaining old ranks:"
  echo "$PIDS"
  kill -TERM $PIDS 2>/dev/null
  for _ in $(seq 1 6); do
    sleep 5
    [ -z "$(find_pids)" ] && break
  done
fi

PIDS=$(find_pids)
if [ -n "$PIDS" ]; then
  echo "Force-clearing unresponsive old ranks:"
  echo "$PIDS"
  kill -KILL $PIDS 2>/dev/null
  sleep 5
fi

PIDS=$(find_pids)
if [ -n "$PIDS" ]; then
  echo "ERROR: old SEAM processes remain: $PIDS"
  STATUS=1
else
  date --iso-8601=seconds > "$RESULTS/STOPPED_FOR_MAX_RESOURCE_RESTART"
  sync
  echo "OLD_SEAM_SUITE_STOPPED=True"
  echo "All previous files and partial results remain unchanged."
  STATUS=0
fi

echo "Stop status: $STATUS"
echo "Terminal remains open."
exit "$STATUS"
