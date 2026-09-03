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
PACKAGE_VERSION=$(tr -d '\r\n[:space:]' < "$PACKAGE/SUITE_VERSION" 2>/dev/null)

bash "$PACKAGE/install_seam_environment.sh"
STATUS=$?

if [ "$STATUS" -eq 0 ] && [ "${SEAM_FORCE_NEW_SUITE:-0}" != "1" ] && \
   [ -s "$POINTER" ] && [ -s "$RESULTS_POINTER" ]; then
  CANDIDATE=$(tr -d '\r\n' < "$POINTER")
  CANDIDATE_RESULTS=$(tr -d '\r\n' < "$RESULTS_POINTER")
  CANDIDATE_VERSION=$(tr -d '\r\n[:space:]' < "$CANDIDATE/SUITE_VERSION" 2>/dev/null)
  if [ -x "$CANDIDATE/run_seam_20epoch_4test_suite.sh" ] && \
     [ -d "$CANDIDATE_RESULTS" ]; then
    if [ -n "$PACKAGE_VERSION" ] && [ "$CANDIDATE_VERSION" = "$PACKAGE_VERSION" ]; then
      EXPERIMENT_ROOT="$CANDIDATE"
      RESULTS_ROOT="$CANDIDATE_RESULTS"
      echo "REUSING_RESUMABLE_EXPERIMENT=$EXPERIMENT_ROOT"
    elif find "$CANDIDATE_RESULTS" -type f -name '*.ckpt' -print -quit \
        2>/dev/null | grep -q .; then
      echo "ERROR: the resumable experiment uses suite version ${CANDIDATE_VERSION:-legacy},"
      echo "but the downloaded package uses version ${PACKAGE_VERSION:-unknown}."
      echo "Existing checkpoints were preserved; no incompatible code was applied."
      STATUS=1
    else
      echo "PRESERVING_PREFLIGHT_ONLY_EXPERIMENT=$CANDIDATE"
      echo "CREATING_CORRECTED_SUITE_VERSION=${PACKAGE_VERSION:-unknown}"
    fi
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
