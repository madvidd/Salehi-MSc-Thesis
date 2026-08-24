#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

BASE=${BASE:-/home/server01/M}
PACKAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POINTER="$BASE/Codes/LATEST_SEAM_AV2_MAMBA_3RUN_CODE.txt"
EXPERIMENT_ROOT=""
STATUS=0

bash "$PACKAGE/install_seam_environment.sh"
STATUS=$?

if [ "$STATUS" -eq 0 ] && [ -s "$POINTER" ]; then
  CANDIDATE=$(tr -d '\r\n' < "$POINTER")
  RESULTS_POINTER="$BASE/Results/LATEST_SEAM_AV2_MAMBA_3RUN.txt"
  if [ -x "$CANDIDATE/run_seam_3run_suite.sh" ] && \
     [ -s "$RESULTS_POINTER" ]; then
    CANDIDATE_RESULTS=$(tr -d '\r\n' < "$RESULTS_POINTER")
    if [ ! -f "$CANDIDATE_RESULTS/SUITE_COMPLETE" ]; then
      EXPERIMENT_ROOT="$CANDIDATE"
      echo "REUSING_RESUMABLE_EXPERIMENT=$EXPERIMENT_ROOT"
    fi
  fi
fi

if [ "$STATUS" -eq 0 ] && [ -z "$EXPERIMENT_ROOT" ]; then
  python3 "$PACKAGE/setup_lab3_seam_3run.py" --base "$BASE"
  STATUS=$?
  if [ "$STATUS" -eq 0 ]; then
    EXPERIMENT_ROOT=$(tr -d '\r\n' < "$POINTER")
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  bash "$EXPERIMENT_ROOT/run_seam_3run_suite.sh"
  STATUS=$?
fi

echo
echo "Foreground SEAM suite status: $STATUS"
echo "Terminal remains open."
exit "$STATUS"
