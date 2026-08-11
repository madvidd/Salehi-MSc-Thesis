#!/usr/bin/env bash

# Publish a read-only snapshot of the active Lab 2 ten-test suite. This script
# never sends signals to, suspends, or otherwise controls training processes.

set -uo pipefail

BASE=/home/server00/M
TOKEN_FILE="$BASE/Token/Token.txt"
POINTER="$BASE/Results/LATEST_SHARP_AV2_20EPOCH_10TEST.txt"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAMP=$(date +%Y%m%d-%H%M%S)
GIT=/usr/bin/git
STATUS=1

if [[ ! -s "$POINTER" ]]; then
  echo "FATAL: result pointer is missing: $POINTER"
  exit 1
fi
RESULTS=$(tr -d '\r\n' < "$POINTER")
SUITE_LOG="$RESULTS/suite.log"
RUN_NAME=$(basename "$RESULTS")
STAGE="$BASE/Results/Lab2_10Test_Snapshot_Staging/$STAMP"
CLONE="$BASE/Codes/Thesis_Lab2_10Test_Snapshot_$STAMP"
REL="Lab 2/SHARP_20_Epoch_10_Test_Suite/Runs/$RUN_NAME"
LOCAL_DIR="$BASE/Terminal/SHARP_20_Epoch_10_Test_Suite/$RUN_NAME"
LOCAL_TERMINAL="$LOCAL_DIR/Terminal.txt"

if [[ ! -s "$SUITE_LOG" ]]; then
  echo "FATAL: suite log was not found: $SUITE_LOG"
  exit 1
fi
if [[ ! -s "$TOKEN_FILE" ]]; then
  echo "FATAL: token file was not found: $TOKEN_FILE"
  exit 1
fi

mkdir -p "$STAGE" "$LOCAL_DIR"
echo "Saving the complete current suite log locally..."
cp --reflink=auto "$SUITE_LOG" "$LOCAL_TERMINAL" || exit 1
sync "$LOCAL_TERMINAL" 2>/dev/null || sync

"${PYTHON:-python3}" "$SCRIPT_DIR/generate_lab2_10test_snapshot.py" \
  "$RESULTS" "$SUITE_LOG" "$STAGE" || exit 1

find "$RESULTS" -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' | sort \
  > "$STAGE/CHECKPOINTS.txt"
{
  echo "# Lab 2 SHARP 20-Epoch Ten-Test Run"
  echo
  echo "- Captured: $(date --iso-8601=seconds)"
  echo "- Results: $RESULTS"
  echo "- Full local log: $LOCAL_TERMINAL"
  echo "- GitHub Terminal.txt is compacted to remove repeated progress refreshes."
  echo "- Checkpoints and oversized raw logs remain on Lab 2."
  echo "- Training was not interrupted."
} > "$STAGE/README.md"

if find "$STAGE" -type f -size +25M | grep -q .; then
  echo "FATAL: publication staging contains a file larger than 25 MiB."
  exit 1
fi
if grep -RIlE 'github_pat_|ghp_' "$STAGE" >/dev/null 2>&1; then
  echo "FATAL: credential text was detected in publication staging."
  exit 1
fi

TOKEN=$(python3 - "$TOKEN_FILE" <<'PY'
import pathlib
import re
import sys

data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
PY
)
if [[ -z "$TOKEN" ]]; then
  echo "FATAL: no GitHub PAT was found in $TOKEN_FILE"
  exit 1
fi
LOGIN=$(curl -fsS -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/user 2>/dev/null | \
  python3 -c 'import json,sys; print(json.load(sys.stdin).get("login",""))' \
  2>/dev/null)
PUSH=$(curl -fsS -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/repos/madvidd/Thesis 2>/dev/null | \
  python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("permissions",{}).get("push",False)).lower())' \
  2>/dev/null)
echo "GitHub account: $LOGIN"
echo "Push permission: $PUSH"
if [[ "$LOGIN" != madviddd || "$PUSH" != true ]]; then
  echo "FATAL: Token.txt is not a writable madviddd token."
  exit 1
fi

export LAB2_GITHUB_TOKEN="$TOKEN"
ASKPASS=$(mktemp)
printf '%s\n' '#!/usr/bin/env bash' \
  'case "$1" in' \
  '*Username*) printf "%s\n" "madviddd" ;;' \
  '*Password*) printf "%s\n" "$LAB2_GITHUB_TOKEN" ;;' \
  'esac' > "$ASKPASS"
chmod 700 "$ASKPASS"

unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY
unset GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR GIT_PREFIX
unset GIT_CEILING_DIRECTORIES GIT_DISCOVERY_ACROSS_FILESYSTEM
unset GIT_CONFIG_PARAMETERS GIT_EXEC_PATH GIT_TEMPLATE_DIR

GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
  "$GIT" -c credential.helper= clone --depth 1 --branch main --single-branch \
  https://github.com/madvidd/Thesis.git "$CLONE"
STATUS=$?

if (( STATUS == 0 )); then
  mkdir -p "$CLONE/$REL"
  cp -p "$STAGE"/* "$CLONE/$REL/"
  "$GIT" -C "$CLONE" config user.name madviddd
  "$GIT" -C "$CLONE" config user.email madviddd@users.noreply.github.com
  "$GIT" -C "$CLONE" config pull.rebase false
  "$GIT" -C "$CLONE" config merge.autoStash true
  "$GIT" -C "$CLONE" add -- "$REL"
  if ! "$GIT" -C "$CLONE" diff --cached --quiet; then
    "$GIT" -C "$CLONE" commit -m "Update Lab 2 ten-test progress snapshot"
    STATUS=$?
  fi
fi

if (( STATUS == 0 )); then
  for attempt in 1 2 3; do
    GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
      "$GIT" -C "$CLONE" -c credential.helper= \
      pull --no-rebase origin main || {
        STATUS=$?
        break
      }
    GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
      "$GIT" -C "$CLONE" -c credential.helper= push origin main && {
        STATUS=0
        break
      }
    STATUS=$?
    sleep $(( attempt * 3 ))
  done
fi

rm -f "$ASKPASS"
unset LAB2_GITHUB_TOKEN TOKEN
echo
echo "GitHub path: $REL"
echo "Full local log: $LOCAL_TERMINAL"
echo "Summary: $REL/Summary.md"
echo "Publish status: $STATUS"
echo "Training was not interrupted."
echo "Terminal remains open."
exit "$STATUS"
