#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

BASE=${BASE:-/home/server01/M}
PACKAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POINTER="$BASE/Codes/LATEST_SEAM_AV2_20EPOCH_4TEST_MAXRES_CODE.txt"
RESULTS_POINTER="$BASE/Results/LATEST_SEAM_AV2_20EPOCH_4TEST_MAXRES.txt"
EXPERIMENT_ROOT=""
RESULTS_ROOT=""
STATUS=0
PACKAGE_VERSION=$(tr -d '\r\n[:space:]' < "$PACKAGE/MAX_RESOURCE_SUITE_VERSION" 2>/dev/null)

bash "$PACKAGE/install_seam_environment.sh"
STATUS=$?

if [ "$STATUS" -eq 0 ] && [ "${SEAM_FORCE_NEW_MAX_SUITE:-0}" != "1" ] && \
   [ -s "$POINTER" ] && [ -s "$RESULTS_POINTER" ]; then
  CANDIDATE=$(tr -d '\r\n' < "$POINTER")
  CANDIDATE_RESULTS=$(tr -d '\r\n' < "$RESULTS_POINTER")
  CANDIDATE_VERSION=$(tr -d '\r\n[:space:]' < "$CANDIDATE/SUITE_VERSION" 2>/dev/null)
  if [ -x "$CANDIDATE/run_seam_20epoch_4test_suite.sh" ] && \
     [ -d "$CANDIDATE_RESULTS" ]; then
    if [ -n "$PACKAGE_VERSION" ] && [ "$CANDIDATE_VERSION" = "$PACKAGE_VERSION" ]; then
      EXPERIMENT_ROOT="$CANDIDATE"
      RESULTS_ROOT="$CANDIDATE_RESULTS"
      echo "REUSING_RESUMABLE_MAX_RESOURCE_EXPERIMENT=$EXPERIMENT_ROOT"
    elif find "$CANDIDATE_RESULTS" -type f -name '*.ckpt' -print -quit \
        2>/dev/null | grep -q .; then
      echo "ERROR: existing max-resource checkpoints use an incompatible suite version."
      echo "The checkpoints were preserved and no code was replaced."
      STATUS=1
    fi
  fi
fi

if [ "$STATUS" -eq 0 ] && [ -z "$EXPERIMENT_ROOT" ]; then
  python3 "$PACKAGE/setup_lab3_seam_20epoch_4test_max_resources.py" --base "$BASE"
  STATUS=$?
  if [ "$STATUS" -eq 0 ]; then
    EXPERIMENT_ROOT=$(tr -d '\r\n' < "$POINTER")
    RESULTS_ROOT=$(tr -d '\r\n' < "$RESULTS_POINTER")
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  CONFLICT=$(ps -u "$USER" -o pid=,args= | awk -v current="$RESULTS_ROOT" '
    /train[.]py/ && /SEAM_AV2_20EPOCH_4TEST_/ && index($0, current) == 0 {print $1}
  ')
  if [ -n "$CONFLICT" ]; then
    echo "ERROR: another SEAM four-test process still owns the GPUs: $CONFLICT"
    echo "Run stop_current_seam_20epoch_suite.sh first."
    STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  mkdir -p "$RESULTS_ROOT"
  SEAM_GPU_IDS=0,1,2 \
    bash "$EXPERIMENT_ROOT/run_seam_20epoch_4test_suite.sh" \
      2>&1 | tee -a "$RESULTS_ROOT/suite.log"
  STATUS=${PIPESTATUS[0]}
fi

echo
echo "Foreground max-resource SEAM suite status: $STATUS"
echo "Relaunching this command resumes any incomplete variant."
echo "Terminal remains open."
exit "$STATUS"
