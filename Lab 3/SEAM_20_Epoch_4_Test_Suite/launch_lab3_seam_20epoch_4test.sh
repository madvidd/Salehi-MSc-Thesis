#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

BASE=${BASE:-/home/server01/M}
PACKAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POINTER="$BASE/Codes/LATEST_SEAM_AV2_20EPOCH_4TEST_CODE.txt"
RESULTS_POINTER="$BASE/Results/LATEST_SEAM_AV2_20EPOCH_4TEST.txt"
EXPERIMENT_ROOT=""
RESULTS_ROOT=""
STATUS=0

bash "$PACKAGE/install_seam_environment.sh"
STATUS=$?

if [ "$STATUS" -eq 0 ] && [ "${SEAM_FORCE_NEW_SUITE:-0}" != "1" ] && \
   [ -s "$POINTER" ] && [ -s "$RESULTS_POINTER" ]; then
  CANDIDATE=$(tr -d '\r\n' < "$POINTER")
  CANDIDATE_RESULTS=$(tr -d '\r\n' < "$RESULTS_POINTER")
  if [ -x "$CANDIDATE/run_seam_20epoch_4test_suite.sh" ] && \
     [ -d "$CANDIDATE_RESULTS" ]; then
    EXPERIMENT_ROOT="$CANDIDATE"
    RESULTS_ROOT="$CANDIDATE_RESULTS"
    echo "REUSING_RESUMABLE_EXPERIMENT=$EXPERIMENT_ROOT"
  fi
fi

if [ "$STATUS" -eq 0 ] && [ -z "$EXPERIMENT_ROOT" ]; then
  python3 "$PACKAGE/setup_lab3_seam_20epoch_4test.py" --base "$BASE"
  STATUS=$?
  if [ "$STATUS" -eq 0 ]; then
    EXPERIMENT_ROOT=$(tr -d '\r\n' < "$POINTER")
    RESULTS_ROOT=$(tr -d '\r\n' < "$RESULTS_POINTER")
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  mkdir -p "$RESULTS_ROOT"
  bash "$EXPERIMENT_ROOT/run_seam_20epoch_4test_suite.sh" \
    2>&1 | tee -a "$RESULTS_ROOT/suite.log"
  STATUS=${PIPESTATUS[0]}
fi

echo
echo "Foreground SEAM four-test suite status: $STATUS"
echo "Relaunching this same command resumes any incomplete variant."
echo "Terminal remains open."
exit "$STATUS"
