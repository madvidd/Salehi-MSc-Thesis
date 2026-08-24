#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

VARIANT=$1
VARIANT_DIR=$2
RESULTS_ROOT=$3
EXPERIMENT_ROOT=$4
BASE=${BASE:-/home/server01/M}
TOKEN_FILE="$BASE/Token/Token.txt"
GIT=/usr/bin/git
STAMP=$(date +%Y%m%d-%H%M%S)
CLONE="$BASE/Codes/Thesis_SEAM_Publish_${STAMP}_$$"
RUN_NAME=$(basename "$RESULTS_ROOT")
REL="Lab 3/SEAM_AV2_Mamba_3_Run/Results/$RUN_NAME/$VARIANT"
STATUS=1

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

if [ "$LOGIN" != "madviddd" ] || [ "$PUSH" != "true" ]; then
  echo "WARNING: result publication skipped because Token.txt is not a writable madviddd PAT."
  exit 1
fi

export SEAM_GITHUB_TOKEN="$TOKEN"
unset TOKEN
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
  DEST="$CLONE/$REL"
  mkdir -p "$DEST/config"

  for file in Summary.md STATUS.txt CHECKPOINTS.txt LOG_TAIL.txt; do
    [ -f "$VARIANT_DIR/$file" ] && cp -p "$VARIANT_DIR/$file" "$DEST/"
  done
  cp -p "$EXPERIMENT_ROOT/RUN_MANIFEST.json" "$DEST/" 2>/dev/null
  cp -p "$EXPERIMENT_ROOT/Code/conf/config.yaml" "$DEST/config/" 2>/dev/null
  cp -p "$EXPERIMENT_ROOT/Code/conf/model/Seam.yaml" "$DEST/config/" 2>/dev/null
  cp -p "$EXPERIMENT_ROOT/Code/conf/datamodule/av2_stream.yaml" "$DEST/config/" 2>/dev/null

  printf '%s\n' '*.ckpt' '*.tar.gz' '*.pt' 'train.log' 'Token.txt' > "$DEST/.gitignore"

  LARGE=$(find "$DEST" -type f -size +10M -print)
  CREDENTIALS=$(grep -RIlE 'github_pat_|ghp_[A-Za-z0-9]+' "$DEST" 2>/dev/null)
  if [ -n "$LARGE" ] || [ -n "$CREDENTIALS" ]; then
    echo "ERROR: publication staging failed size or credential checks."
    STATUS=1
  else
    "$GIT" -C "$CLONE" config user.name madviddd
    "$GIT" -C "$CLONE" config user.email madviddd@users.noreply.github.com
    "$GIT" -C "$CLONE" config pull.rebase false
    "$GIT" -C "$CLONE" config merge.autoStash true
    "$GIT" -C "$CLONE" add -- "$REL"

    if "$GIT" -C "$CLONE" diff --cached --quiet; then
      echo "No new result changes required a commit."
      STATUS=0
    else
      "$GIT" -C "$CLONE" commit -m "Update SEAM AV2 $VARIANT results"
      STATUS=$?
    fi

    ATTEMPT=1
    while [ "$STATUS" -eq 0 ] && [ "$ATTEMPT" -le 3 ]; do
      GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
        "$GIT" -C "$CLONE" -c credential.helper= pull --no-rebase origin main
      STATUS=$?
      if [ "$STATUS" -eq 0 ]; then
        GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
          "$GIT" -C "$CLONE" -c credential.helper= push origin main
        STATUS=$?
      fi
      [ "$STATUS" -eq 0 ] && break
      ATTEMPT=$((ATTEMPT + 1))
      sleep 5
    done
  fi
fi

rm -f "$ASKPASS"
unset SEAM_GITHUB_TOKEN

if [ "$STATUS" -eq 0 ]; then
  echo "SEAM_RESULT_PUBLISHED=$REL"
else
  echo "WARNING: local results are safe, but GitHub publication did not complete."
fi
exit "$STATUS"
