#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

BASE=${BASE:-/home/server01/M}
EXPERIMENT_ROOT=${1:-$(tr -d '\r\n' < "$BASE/Codes/LATEST_SEAM_AV2_20EPOCH_4TEST_CODE.txt" 2>/dev/null)}
RESULTS_ROOT=${2:-$(tr -d '\r\n' < "$BASE/Results/LATEST_SEAM_AV2_20EPOCH_4TEST.txt" 2>/dev/null)}
PHASE=${3:-progress}
TOKEN_FILE="$BASE/Token/Token.txt"
PYTHON="$BASE/Codes/envs/seam_av2_mamba_torch211/bin/python"
GIT=/usr/bin/git
STAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME=$(basename "$RESULTS_ROOT")
LOCAL="$BASE/Terminal/SEAM_20_Epoch_4_Test_Suite/$RUN_NAME"
STAGE="$BASE/Results/SEAM_20_Epoch_4_Test_Suite_Snapshots/$RUN_NAME/$STAMP"
CLONE="$BASE/Codes/Thesis_SEAM20_Publish_${STAMP}_$$"
REL="Studies/SEAM/Context_Attention_Ablation/Results/$RUN_NAME"
STATUS=1

select_writable_token() {
  local token_file=$1 candidate login push
  while IFS= read -r candidate; do
    [ -n "$candidate" ] || continue
    login=$(curl -fsSL -H "Authorization: Bearer $candidate" \
      https://api.github.com/user 2>/dev/null | \
      python3 -c 'import json,sys; print(json.load(sys.stdin).get("login",""))' 2>/dev/null)
    push=$(curl -fsSL -H "Authorization: Bearer $candidate" \
      https://api.github.com/repos/madvidd/Thesis 2>/dev/null | \
      python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("permissions",{}).get("push",False)).lower())' 2>/dev/null)
    if [ "$login" = "madviddd" ] && [ "$push" = "true" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(
    python3 - "$token_file" "${LAB3_GITHUB_TOKEN:-}" <<'PY'
from pathlib import Path
import re
import sys

values = list(sys.argv[2:])
path = Path(sys.argv[1])
if path.is_file():
    data = path.read_bytes()
    values.extend((data.decode("utf-8-sig", "ignore"), data.decode("utf-16", "ignore")))
seen = set()
for value in values:
    for token in re.findall(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", value):
        if token not in seen:
            print(token)
            seen.add(token)
PY
  )
  return 1
}

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: verified SEAM environment is missing: $PYTHON"
elif [ ! -d "$EXPERIMENT_ROOT" ] || [ ! -d "$RESULTS_ROOT" ]; then
  echo "ERROR: experiment or results directory is missing."
else
  "$PYTHON" "$EXPERIMENT_ROOT/generate_seam_20epoch_summary.py" \
    --results-root "$RESULTS_ROOT"
  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    "$PYTHON" "$EXPERIMENT_ROOT/generate_seam_20epoch_snapshot.py" \
      --results-root "$RESULTS_ROOT" \
      --experiment-root "$EXPERIMENT_ROOT" \
      --local-dir "$LOCAL" \
      --stage-dir "$STAGE"
    STATUS=$?
  fi

  TOKEN=$(select_writable_token "$TOKEN_FILE")
  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: local result-summary generation failed."
  elif [ -z "$TOKEN" ]; then
    echo "ERROR: no writable madviddd token was found in $TOKEN_FILE"
    STATUS=1
  else
    LOGIN=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
      https://api.github.com/user 2>/dev/null | \
      python3 -c 'import json,sys; print(json.load(sys.stdin).get("login",""))' 2>/dev/null)
    PUSH=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
      https://api.github.com/repos/madvidd/Thesis 2>/dev/null | \
      python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("permissions",{}).get("push",False)).lower())' 2>/dev/null)
    echo "GitHub account: $LOGIN"
    echo "Push permission: $PUSH"

    LARGE=$(find "$STAGE" -type f -size +10M -print)
    CREDENTIALS=$(grep -RIlE 'github_pat_|ghp_[A-Za-z0-9]+' "$STAGE" 2>/dev/null)
    if [ "$LOGIN" != "madviddd" ] || [ "$PUSH" != "true" ] || \
       [ -n "$LARGE" ] || [ -n "$CREDENTIALS" ]; then
      echo "ERROR: authentication, size, or credential safety check failed."
      STATUS=1
    else
      export LAB3_GITHUB_TOKEN="$TOKEN"
      ASKPASS=$(mktemp)
      printf '%s\n' \
        '#!/usr/bin/env bash' \
        'case "$1" in' \
        ' *Username*) printf "%s\n" "madviddd" ;;' \
        ' *Password*) printf "%s\n" "$LAB3_GITHUB_TOKEN" ;;' \
        'esac' > "$ASKPASS"
      chmod 700 "$ASKPASS"

      unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_EXEC_PATH GIT_TEMPLATE_DIR
      GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
        "$GIT" -c credential.helper= clone --depth 1 --branch main \
        https://github.com/madvidd/Thesis.git "$CLONE"
      STATUS=$?

      if [ "$STATUS" -eq 0 ]; then
        mkdir -p "$CLONE/$REL"
        cp -a "$STAGE/." "$CLONE/$REL/"

        LARGE=$(find "$CLONE/$REL" -type f -size +10M -print)
        CREDENTIALS=$(grep -RIlE 'github_pat_|ghp_[A-Za-z0-9]+' "$CLONE/$REL" 2>/dev/null)
        if [ -n "$LARGE" ] || [ -n "$CREDENTIALS" ]; then
          echo "ERROR: merged publication folder failed safety checks."
          STATUS=1
        fi
      fi

      if [ "$STATUS" -eq 0 ]; then
        "$GIT" -C "$CLONE" config user.name madviddd
        "$GIT" -C "$CLONE" config user.email madviddd@users.noreply.github.com
        "$GIT" -C "$CLONE" config pull.rebase false
        "$GIT" -C "$CLONE" config merge.autoStash true
        "$GIT" -C "$CLONE" add -- "$REL"
        STATUS=$?

        if [ "$STATUS" -ne 0 ]; then
          echo "ERROR: Git refused to stage the curated snapshot."
        elif ! "$GIT" -C "$CLONE" ls-files --error-unmatch \
            "$REL/Summary.md" >/dev/null 2>&1 || \
             ! "$GIT" -C "$CLONE" ls-files --error-unmatch \
            "$REL/Terminal.txt" >/dev/null 2>&1; then
          echo "ERROR: required snapshot files were not added to the Git index."
          STATUS=1
        elif "$GIT" -C "$CLONE" diff --cached --quiet; then
          echo "No new snapshot changes required a commit."
        else
          "$GIT" -C "$CLONE" commit -m "Update SEAM 20-epoch four-test $PHASE"
          STATUS=$?
        fi
      fi

      if [ "$STATUS" -eq 0 ]; then
        STATUS=1
        for ATTEMPT in 1 2 3; do
          GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
            "$GIT" -C "$CLONE" -c credential.helper= pull --no-rebase origin main
          PULL_STATUS=$?
          if [ "$PULL_STATUS" -eq 0 ]; then
            GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
              "$GIT" -C "$CLONE" -c credential.helper= push origin main
            STATUS=$?
          else
            STATUS=$PULL_STATUS
          fi
          [ "$STATUS" -eq 0 ] && break
          echo "Publication attempt $ATTEMPT failed; automatically merging and retrying."
          sleep 10
        done
      fi

      if [ "$STATUS" -eq 0 ]; then
        GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
          "$GIT" -C "$CLONE" -c credential.helper= fetch origin main
        STATUS=$?
      fi

      if [ "$STATUS" -eq 0 ]; then
        if "$GIT" -C "$CLONE" cat-file -e \
             "origin/main:$REL/Summary.md" 2>/dev/null && \
           "$GIT" -C "$CLONE" cat-file -e \
             "origin/main:$REL/Terminal.txt" 2>/dev/null; then
          echo "REMOTE_SNAPSHOT_VERIFIED=True"
        else
          echo "ERROR: required snapshot files are absent from remote main."
          STATUS=1
        fi
      fi

      if [ "$STATUS" -eq 0 ]; then
        printf '%s\n' "$(date --iso-8601=seconds)" > "$RESULTS_ROOT/LAST_PUBLISHED_AT.txt"
        echo "SEAM20_SNAPSHOT_PUBLISHED=$REL"
      fi
      rm -f "$ASKPASS"
      unset LAB3_GITHUB_TOKEN
    fi
  fi
  unset TOKEN
fi

echo "Publication status: $STATUS"
echo "Training was not interrupted."
echo "Terminal remains open."
exit "$STATUS"
