#!/usr/bin/env bash
set -uo pipefail

BASE=/home/server00/M
PACKAGE="$BASE/Codes/Thesis/Lab 2/SHARP_Final_3_Run_Suite"
POINTER="$BASE/Results/LATEST_SHARP_FINAL_3RUN_RESULTS.txt"
TOKEN_FILE="$BASE/Token/Token.txt"
PYTHON_BIN="$BASE/Codes/envs/sharp/bin/python"
GIT=/usr/bin/git
STATUS=1

RESULTS_ROOT=$(tr -d '\r\n' < "$POINTER" 2>/dev/null)
if [ -z "$RESULTS_ROOT" ] || [ ! -d "$RESULTS_ROOT" ]; then
  echo "ERROR: final-suite results pointer is unavailable: $POINTER"
  return 1 2>/dev/null || true
else
  SUITE_NAME=$(basename "$RESULTS_ROOT")
  STAMP=$(date +%Y%m%d-%H%M%S)
  LOCAL_DIR="$BASE/Terminal/SHARP_Final_3_Run_Suite/$SUITE_NAME/Progress_$STAMP"
  STAGE="$BASE/Results/SHARP_Final_3_Run_Suite_Snapshots/$SUITE_NAME/$STAMP"
  CLONE="$BASE/Codes/Thesis_Final3_Progress_Publish_$STAMP"
  REL="Lab 2/SHARP_Final_3_Run_Suite/Results/$SUITE_NAME/Current_Progress"
  mkdir -p "$LOCAL_DIR" "$STAGE"

  echo "Saving a complete one-time suite transcript locally..."
  ionice -c3 nice -n 19 "$PYTHON_BIN" \
    "$PACKAGE/generate_progress_snapshot.py" \
    --results "$RESULTS_ROOT" --local-dir "$LOCAL_DIR" --stage "$STAGE"
  STATUS=$?

  LARGE=$(find "$STAGE" -type f -size +10M -print 2>/dev/null)
  CREDENTIALS=$(grep -RIlE 'github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+' \
    "$STAGE" 2>/dev/null)
  if [ "$STATUS" -ne 0 ] || [ -n "$LARGE" ] || [ -n "$CREDENTIALS" ]; then
    echo "ERROR: snapshot generation, size check, or credential scan failed."
    STATUS=1
  else
    TOKEN=$("$PYTHON_BIN" - "$TOKEN_FILE" <<'PY'
import pathlib
import re
import sys

data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
PY
)
    LOGIN=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
      https://api.github.com/user 2>/dev/null | \
      "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("login", ""))' 2>/dev/null)
    PUSH=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
      https://api.github.com/repos/madvidd/Thesis 2>/dev/null | \
      "$PYTHON_BIN" -c 'import json,sys; print(str(json.load(sys.stdin).get("permissions", {}).get("push", False)).lower())' 2>/dev/null)
    echo "GitHub account: $LOGIN"
    echo "Push permission: $PUSH"

    if [ "$LOGIN" != "madviddd" ] || [ "$PUSH" != "true" ]; then
      echo "ERROR: Token.txt is not a writable madviddd token."
      STATUS=1
    else
      export FINAL3_GITHUB_TOKEN="$TOKEN"
      ASKPASS=$(mktemp)
      printf '%s\n' \
        '#!/usr/bin/env bash' \
        'case "$1" in' \
        '  *Username*) printf "%s\n" "madviddd" ;;' \
        '  *Password*) printf "%s\n" "$FINAL3_GITHUB_TOKEN" ;;' \
        'esac' > "$ASKPASS"
      chmod 700 "$ASKPASS"

      unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY
      unset GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_COMMON_DIR GIT_PREFIX
      unset GIT_CEILING_DIRECTORIES GIT_EXEC_PATH GIT_TEMPLATE_DIR

      GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
        "$GIT" -c credential.helper= clone --depth 1 --single-branch \
        --branch main https://github.com/madvidd/Thesis.git "$CLONE"
      STATUS=$?
      if [ "$STATUS" -eq 0 ]; then
        mkdir -p "$CLONE/$REL"
        cp -a "$STAGE/." "$CLONE/$REL/"
        "$GIT" -C "$CLONE" config user.name madviddd
        "$GIT" -C "$CLONE" config user.email madviddd@users.noreply.github.com
        "$GIT" -C "$CLONE" config pull.rebase false
        "$GIT" -C "$CLONE" config merge.autoStash true
        "$GIT" -C "$CLONE" add -- "$REL"
        if ! "$GIT" -C "$CLONE" diff --cached --quiet; then
          "$GIT" -C "$CLONE" commit -m "Update Lab 2 final-suite progress $STAMP"
          STATUS=$?
        fi
      fi

      if [ "$STATUS" -eq 0 ]; then
        STATUS=1
        for ATTEMPT in 1 2 3; do
          GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
            "$GIT" -C "$CLONE" -c credential.helper= pull --no-rebase origin main
          STATUS=$?
          [ "$STATUS" -ne 0 ] && break
          GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
            "$GIT" -C "$CLONE" -c credential.helper= push origin main
          STATUS=$?
          [ "$STATUS" -eq 0 ] && break
          echo "Push attempt $ATTEMPT failed; automatically merging and retrying."
          sleep 10
        done
      fi

      if [ "$STATUS" -eq 0 ]; then
        echo "LAB2_FINAL3_PROGRESS_PUBLISHED"
        echo "Published commit: $("$GIT" -C "$CLONE" rev-parse HEAD)"
        echo "GitHub path: $REL"
      else
        echo "ERROR: automatic pull, merge, or push failed."
      fi
      rm -f "$ASKPASS"
      unset FINAL3_GITHUB_TOKEN
    fi
    unset TOKEN
  fi
fi

echo "Training was not interrupted."
echo "Publish status: $STATUS"
echo "Terminal remains open."
return "$STATUS" 2>/dev/null || true
