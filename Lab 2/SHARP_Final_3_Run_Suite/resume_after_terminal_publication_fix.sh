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

if [ ! -d "$EXPERIMENT_ROOT" ] || [ ! -d "$RESULTS_ROOT" ]; then
  echo "ERROR: the existing final-suite code or results pointer is invalid."
  echo "Code:    $EXPERIMENT_ROOT"
  echo "Results: $RESULTS_ROOT"
  exit 1
fi

ACTIVE=$(
  ps -u "$USER" -o pid=,args= | awk \
    -v code="$EXPERIMENT_ROOT" -v results="$RESULTS_ROOT" '
      (index($0, code) || index($0, results)) &&
      ($0 ~ /[t]rain[.]py/ || $0 ~ /run_final_3run_suite[.]sh/) {print}
    '
)
if [ -n "$ACTIVE" ]; then
  echo "ERROR: this final suite is already active; no duplicate was started."
  echo "$ACTIVE"
  exit 1
fi

RUN1="$RESULTS_ROOT/01_official_sharp_baseline"
RUN1_BEST_FILE="$RUN1/run/BEST_CHECKPOINT.txt"
BEST=$(tr -d '\r\n' < "$RUN1_BEST_FILE" 2>/dev/null || true)

if [ ! -s "$RESULTS_ROOT/suite.log" ]; then
  echo "ERROR: the complete suite log is missing."
  exit 1
fi
if [ ! -s "$RUN1/COMPLETED" ]; then
  echo "ERROR: Run 1 has no valid COMPLETED marker; refusing to skip it."
  exit 1
fi
if [ ! -s "$RUN1/artifacts/metrics.json" ]; then
  echo "ERROR: Run 1 final metrics are missing."
  exit 1
fi
if [ -z "$BEST" ] || [ ! -s "$BEST" ]; then
  echo "ERROR: Run 1 best checkpoint is unavailable: $BEST"
  exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
SUITE_NAME=$(basename "$RESULTS_ROOT")
ARCHIVE="$BASE/Terminal/SHARP_Final_3_Run_Suite/$SUITE_NAME/Before_Run2_$STAMP"
mkdir -p "$ARCHIVE/Run1"

echo "Saving the complete append-only suite transcript locally..."
cp --reflink=auto -p "$RESULTS_ROOT/suite.log" "$ARCHIVE/Terminal.txt"
cp -p "$RUN1/COMPLETED" "$ARCHIVE/Run1/COMPLETED.txt"
cp -p "$RUN1/artifacts/metrics.json" "$ARCHIVE/Run1/metrics.json"
cp -p "$RUN1_BEST_FILE" "$ARCHIVE/Run1/BEST_CHECKPOINT.txt"
cp -p "$RUN1/RUN_TIMING.json" "$ARCHIVE/Run1/" 2>/dev/null || true
cp -p "$RUN1/evaluation.log" "$ARCHIVE/Run1/" 2>/dev/null || true
cp -p "$RUN1/artifacts/Summary.md" "$ARCHIVE/Run1/" 2>/dev/null || true
cp -p "$RUN1/TRAIN_COMMAND.txt" "$ARCHIVE/Run1/" 2>/dev/null || true
cp -p "$RUN1/ATTEMPTS.tsv" "$ARCHIVE/Run1/" 2>/dev/null || true
cp -p "$EXPERIMENT_ROOT/SUITE_MANIFEST.json" "$ARCHIVE/" 2>/dev/null || true
cp -p "$EXPERIMENT_ROOT/PINNED_SOURCE_SHA256.txt" "$ARCHIVE/" 2>/dev/null || true
cp -p "$EXPERIMENT_ROOT/suite.env" "$ARCHIVE/" 2>/dev/null || true

mkdir -p "$ARCHIVE/Run1/artifacts"
while IFS= read -r -d '' FILE; do
  RELATIVE=${FILE#"$RUN1/artifacts/"}
  [ "$RELATIVE" = "Terminal.txt" ] && continue
  mkdir -p "$ARCHIVE/Run1/artifacts/$(dirname "$RELATIVE")"
  cp --reflink=auto -p "$FILE" "$ARCHIVE/Run1/artifacts/$RELATIVE"
done < <(find "$RUN1/artifacts" -type f -print0)

find "$RESULTS_ROOT" -type f -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' | sort \
  > "$ARCHIVE/CHECKPOINTS.txt"
ps -u "$USER" -o pid,ppid,pgid,etime,state,%cpu,%mem,args \
  > "$ARCHIVE/PROCESSES.txt"
sha256sum "$BEST" > "$ARCHIVE/Run1/BEST_CHECKPOINT_SHA256.txt"

cat > "$ARCHIVE/SNAPSHOT_INFO.txt" <<EOF
saved=$(date --iso-8601=seconds)
experiment_root=$EXPERIMENT_ROOT
results_root=$RESULTS_ROOT
run1_completed_marker=$RUN1/COMPLETED
run1_best_checkpoint=$BEST
purpose=Preserve the completed official baseline before retrying bounded publication and continuing Runs 2 and 3.
EOF

sync

if [ ! -s "$ARCHIVE/Terminal.txt" ] ||
   [ ! -s "$ARCHIVE/Run1/metrics.json" ] ||
   [ ! -s "$ARCHIVE/Run1/BEST_CHECKPOINT_SHA256.txt" ]; then
  echo "ERROR: local recovery archive verification failed."
  exit 1
fi

echo
echo "PREVIOUS_RUN_ARCHIVE_VERIFIED=$ARCHIVE"
ls -lh "$ARCHIVE/Terminal.txt"
echo "RUN1_COMPLETION_AND_CHECKPOINT_VERIFIED=YES"
echo "SAFE_TO_CLOSE_PREVIOUS_TERMINAL=YES"
echo
echo "Refreshing the existing experiment's control scripts and resuming the suite..."
echo "Run 1 will be reused; Run 2 and Run 3 will train and publish in sequence."

cd "$PACKAGE" || exit 1
bash "$PACKAGE/launch_final_3run_suite.sh"
STATUS=$?

echo
echo "Recovered final-suite status: $STATUS"
echo "Recovery archive: $ARCHIVE"
echo "The terminal remains open."
exit "$STATUS"
