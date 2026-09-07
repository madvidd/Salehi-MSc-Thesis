#!/usr/bin/env bash
set -eu
BASE=${BASE:-/home/server01/M}
PACKAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON="$BASE/Codes/envs/seam_av2_mamba_torch211/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: the previous SEAM Python environment is missing: $PYTHON"
  exit 1
fi
mkdir -p "$BASE/Codes"
exec 8>"$BASE/Codes/.seam80_combined_setup.lock"
flock -n 8 || { echo "Another combined setup is already running."; exit 1; }
"$PYTHON" "$PACKAGE/setup_combined.py" --base "$BASE"
EXPERIMENT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SEAM_AV2_80EPOCH_COMBINED_CODE.txt")
flock -u 8
exec 8>&-
"$PYTHON" "$EXPERIMENT/run_combined.py" --experiment "$EXPERIMENT"
