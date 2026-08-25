#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

BASE=${BASE:-/home/server01/M}
PACKAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOKEN_FILE="$BASE/Token/Token.txt"
RESULTS_POINTER="$BASE/Results/LATEST_SEAM_AV2_MAMBA_3RUN.txt"
EXPERIMENT_POINTER="$BASE/Codes/LATEST_SEAM_AV2_MAMBA_3RUN_CODE.txt"
GIT=/usr/bin/git
STATUS=1

if [ ! -s "$RESULTS_POINTER" ] || [ ! -s "$EXPERIMENT_POINTER" ]; then
  echo "ERROR: SEAM result or experiment pointer is missing."
else
  RESULTS_ROOT=$(tr -d '\r\n' < "$RESULTS_POINTER")
  EXPERIMENT_ROOT=$(tr -d '\r\n' < "$EXPERIMENT_POINTER")
  RUN_NAME=$(basename "$RESULTS_ROOT")
  LOCAL="$BASE/Terminal/SEAM_AV2_Mamba_3_Run/$RUN_NAME"
  STAMP=$(date +%Y%m%d-%H%M%S)
  STAGE="$BASE/Results/SEAM_AV2_Mamba_3_Run/Snapshot_Staging/$STAMP"
  CLONE="$BASE/Codes/Thesis_SEAM_Snapshot_${STAMP}_$$"
  REL="Lab 3/SEAM_AV2_Mamba_3_Run/Results/$RUN_NAME"

  mkdir -p "$LOCAL" "$STAGE"

  for VARIANT in baseline mamba_agent_add mamba_future_replace; do
    RUN="$RESULTS_ROOT/$VARIANT"
    if [ -d "$RUN" ]; then
      python3 "$EXPERIMENT_ROOT/generate_seam_summary.py" \
        --variant-dir "$RUN" \
        --experiment "$EXPERIMENT_ROOT" \
        --variant "$VARIANT" >/dev/null 2>&1
    fi
  done

  python3 "$PACKAGE/generate_seam_suite_snapshot.py" \
    --results-root "$RESULTS_ROOT" \
    --experiment-root "$EXPERIMENT_ROOT" \
    --output-dir "$LOCAL"
  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    echo "Saving all persistent SEAM training logs locally..."
    {
      echo "SEAM AV2 three-run terminal snapshot"
      echo "Saved: $(date --iso-8601=seconds)"
      echo "Results: $RESULTS_ROOT"
      echo "Experiment: $EXPERIMENT_ROOT"
      echo
      for VARIANT in baseline mamba_agent_add mamba_future_replace; do
        LOG="$RESULTS_ROOT/$VARIANT/train.log"
        echo "===== $VARIANT ====="
        if [ -f "$LOG" ]; then
          tr '\r' '\n' < "$LOG"
        else
          echo "No persistent training log exists yet."
        fi
        echo
      done
    } > "$LOCAL/Terminal.txt"
    sync

    cp -p "$LOCAL/Summary.md" "$STAGE/"
    cp -p "$LOCAL/CHECKPOINTS.txt" "$STAGE/"
    cp -p "$LOCAL/CURRENT_ERRORS.txt" "$STAGE/"
    cp -p "$LOCAL/RUN_STATUS.txt" "$STAGE/"

    FULL_SIZE=$(stat -c %s "$LOCAL/Terminal.txt")
    MAX_GITHUB_TERMINAL=$((10 * 1024 * 1024))
    if [ "$FULL_SIZE" -le "$MAX_GITHUB_TERMINAL" ]; then
      cp -p "$LOCAL/Terminal.txt" "$STAGE/Terminal.txt"
    else
      {
        echo "COMPACT_GITHUB_SNAPSHOT=True"
        echo "Full local Terminal.txt: $LOCAL/Terminal.txt"
        echo "Full local bytes: $FULL_SIZE"
        echo "The first 1 MiB and latest 8 MiB are retained below."
        echo
        head -c $((1 * 1024 * 1024)) "$LOCAL/Terminal.txt"
        echo
        echo "===== OMITTED MIDDLE OF LARGE LOCAL LOG ====="
        echo
        tail -c $((8 * 1024 * 1024)) "$LOCAL/Terminal.txt"
      } > "$STAGE/Terminal.txt"
    fi

    printf '%s\n' '*.ckpt' '*.pt' '*.tar.gz' 'Token.txt' > "$STAGE/.gitignore"
    printf '%s\n' \
      "# SEAM AV2 Three-Run Snapshot" "" \
      "Generated: $STAMP" \
      "Local results: $RESULTS_ROOT" \
      "The complete Terminal.txt remains on Lab 3 when the GitHub copy must be compacted." \
      > "$STAGE/README.md"
  fi

  TOKEN=$(python3 - "$TOKEN_FILE" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    print("")
    raise SystemExit
data = path.read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
PY
  )

  LOGIN=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
    https://api.github.com/user 2>/dev/null | \
    python3 -c 'import json,sys; print(json.load(sys.stdin).get("login",""))' \
    2>/dev/null)
  PUSH=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
    https://api.github.com/repos/madvidd/Thesis 2>/dev/null | \
    python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("permissions",{}).get("push",False)).lower())' \
    2>/dev/null)

  echo "GitHub account: $LOGIN"
  echo "Push permission: $PUSH"

  if [ "$STATUS" -eq 0 ] && [ "$LOGIN" = "madviddd" ] && [ "$PUSH" = "true" ]; then
    LARGE=$(find "$STAGE" -type f -size +11M -print)
    CREDENTIALS=$(grep -RIlE 'github_pat_|ghp_[A-Za-z0-9]+' "$STAGE" 2>/dev/null)
    if [ -n "$LARGE" ] || [ -n "$CREDENTIALS" ]; then
      echo "ERROR: snapshot failed its size or credential scan."
      STATUS=1
    else
      export SEAM_GITHUB_TOKEN="$TOKEN"
      ASKPASS=$(mktemp)
      printf '%s\n' \
        '#!/usr/bin/env bash' \
        'case "$1" in' \
        ' *Username*) printf "%s\n" "madviddd" ;;' \
        ' *Password*) printf "%s\n" "$SEAM_GITHUB_TOKEN" ;;' \
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
        "$GIT" -C "$CLONE" config user.name madviddd
        "$GIT" -C "$CLONE" config user.email madviddd@users.noreply.github.com
        "$GIT" -C "$CLONE" config pull.rebase false
        "$GIT" -C "$CLONE" config merge.autoStash true
        "$GIT" -C "$CLONE" add -- "$REL"

        if "$GIT" -C "$CLONE" diff --cached --quiet; then
          echo "No new snapshot changes required a commit."
        else
          "$GIT" -C "$CLONE" commit -m "Update SEAM AV2 three-run progress"
          STATUS=$?
        fi

        if [ "$STATUS" -eq 0 ]; then
          GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
            "$GIT" -C "$CLONE" -c credential.helper= \
            pull --no-rebase origin main
          STATUS=$?
        fi
        if [ "$STATUS" -eq 0 ]; then
          GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
            "$GIT" -C "$CLONE" -c credential.helper= push origin main
          STATUS=$?
        fi
      fi

      rm -f "$ASKPASS"
      unset SEAM_GITHUB_TOKEN
    fi
  else
    echo "ERROR: snapshot generation or GitHub token verification failed."
    STATUS=1
  fi
  unset TOKEN
fi

echo
if [ "$STATUS" -eq 0 ]; then
  echo "SEAM_SUITE_SNAPSHOT_PUBLISHED"
  echo "GitHub path: $REL"
  echo "Full local Terminal.txt: $LOCAL/Terminal.txt"
  echo "Summary: $REL/Summary.md"
fi
echo "Publication status: $STATUS"
echo "Training was not interrupted."
echo "Terminal remains open."
exit "$STATUS"
