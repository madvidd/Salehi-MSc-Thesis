#!/usr/bin/env bash

# Publish a read-only snapshot of the current Lab 3 attention suite. This
# script never sends signals to training processes.

set -uo pipefail

BASE=/home/server01/M
TOKEN_FILE="$BASE/Token/Token.txt"
RESULTS=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAMP=$(date +%Y%m%d-%H%M%S)
STAGE="$BASE/Results/Lab3_Attention_Snapshot_Staging/$STAMP"
CLONE="$BASE/Codes/Thesis_Lab3_Snapshot_$STAMP"
REL="Lab 3/Main_Results/Attention_Experiments/Runs/$(basename "$RESULTS")"
LOCAL_TERMINAL="$BASE/Terminal/Terminal.txt"
GIT=/usr/bin/git
STATUS=1

LATEST_LOG=$(ls -t "$RESULTS"/remaining_attention_suite_*.log 2>/dev/null | head -1)
if [[ ! -s "$LATEST_LOG" ]]; then
  echo "FATAL: no attention-suite log was found under $RESULTS"
  exit 1
fi

mkdir -p "$STAGE" "$(dirname "$LOCAL_TERMINAL")"
cp --reflink=auto "$LATEST_LOG" "$LOCAL_TERMINAL"
"${PYTHON:-python3}" "$SCRIPT_DIR/generate_lab3_attention_snapshot.py" \
  "$RESULTS" "$STAGE" || exit 1

find "$RESULTS" -name '*.ckpt' \
  -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' | sort \
  > "$STAGE/CHECKPOINTS.txt"
{
  echo "Generated: $(date --iso-8601=seconds)"
  echo "Results: $RESULTS"
  echo "Latest raw log: $LATEST_LOG"
  echo "Full local snapshot: $LOCAL_TERMINAL"
  echo "Training was not interrupted."
} > "$STAGE/README.md"

if find "$STAGE" -type f -size +25M | grep -q .; then
  echo "FATAL: publication staging contains a file larger than 25 MiB."
  exit 1
fi
if grep -RIlE 'github_pat_|ghp_' "$STAGE" >/dev/null 2>&1; then
  echo "FATAL: credential text was detected in publication staging."
  exit 1
fi

TOKEN=$(grep -aoE 'github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+' \
  "$TOKEN_FILE" | head -1)
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

export LAB3_GITHUB_TOKEN="$TOKEN"
ASKPASS=$(mktemp)
printf '%s\n' '#!/usr/bin/env bash' \
  'case "$1" in' \
  '*Username*) printf "%s\n" "madviddd" ;;' \
  '*Password*) printf "%s\n" "$LAB3_GITHUB_TOKEN" ;;' \
  'esac' > "$ASKPASS"
chmod 700 "$ASKPASS"

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
  "$GIT" -C "$CLONE" add -- "$REL"
  if ! "$GIT" -C "$CLONE" diff --cached --quiet; then
    "$GIT" -C "$CLONE" commit -m "Update Lab 3 attention progress snapshot"
    STATUS=$?
  fi
fi

if (( STATUS == 0 )); then
  for attempt in 1 2 3; do
    GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
      "$GIT" -C "$CLONE" -c credential.helper= pull --no-rebase origin main || {
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
unset LAB3_GITHUB_TOKEN TOKEN
echo "GitHub path: $REL"
echo "Full local log: $LOCAL_TERMINAL"
echo "Publish status: $STATUS"
echo "Training was not interrupted."
echo "Terminal remains open."
exit "$STATUS"
