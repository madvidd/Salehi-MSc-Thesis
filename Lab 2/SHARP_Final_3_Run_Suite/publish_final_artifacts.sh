#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/suite.env"

SLUG=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --slug) SLUG="${2:-}"; shift 2 ;;
    *) echo "ERROR: unknown publisher argument: $1"; exit 2 ;;
  esac
done

STATUS=1
TOKEN_FILE="$BASE/Token/Token.txt"
SOURCE="$RESULTS_ROOT/$SLUG/artifacts"
COMPARISON="$RESULTS_ROOT/dissertation_artifacts"
STAMP=$(date +%Y%m%d-%H%M%S)
CLONE="$BASE/Codes/Thesis_Final3_Publish_${STAMP}"
SUITE_NAME=$(basename "$RESULTS_ROOT")
REL="Lab 2/SHARP_Final_3_Run_Suite/Results/$SUITE_NAME/$SLUG"
COMPARISON_REL="Lab 2/SHARP_Final_3_Run_Suite/Results/$SUITE_NAME/Comparison"
GIT=/usr/bin/git

if [ -z "$SLUG" ] || [ ! -d "$SOURCE" ]; then
  echo "ERROR: publication source is unavailable: $SOURCE"
  exit 1
fi

TOKEN=$(
  "$BASE/Codes/envs/sharp/bin/python" - "$TOKEN_FILE" <<'PY'
import pathlib
import re
import sys

data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
PY
)

if [ -z "$TOKEN" ]; then
  echo "ERROR: no GitHub PAT found in $TOKEN_FILE"
  exit 1
fi

LOGIN=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/user 2>/dev/null | \
  "$BASE/Codes/envs/sharp/bin/python" -c \
  'import json,sys; print(json.load(sys.stdin).get("login", ""))' 2>/dev/null)
PUSH=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/repos/madvidd/Thesis 2>/dev/null | \
  "$BASE/Codes/envs/sharp/bin/python" -c \
  'import json,sys; print(str(json.load(sys.stdin).get("permissions", {}).get("push", False)).lower())' 2>/dev/null)

echo "GitHub account: $LOGIN"
echo "Push permission: $PUSH"
if [ "$LOGIN" != "madviddd" ] || [ "$PUSH" != "true" ]; then
  echo "ERROR: Token.txt is not a writable madviddd token for madvidd/Thesis."
  unset TOKEN
  exit 1
fi

export FINAL3_GITHUB_TOKEN="$TOKEN"
unset TOKEN
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
  "$GIT" -c credential.helper= clone \
  --depth 1 --single-branch --branch main \
  https://github.com/madvidd/Thesis.git "$CLONE"
STATUS=$?

if [ "$STATUS" -eq 0 ]; then
  mkdir -p "$CLONE/$REL" "$CLONE/$COMPARISON_REL"
  cp -a "$SOURCE/." "$CLONE/$REL/"
  if [ -d "$COMPARISON" ]; then
    cp -a "$COMPARISON/." "$CLONE/$COMPARISON_REL/"
  fi
  printf '%s\n' '*.ckpt' '*.tar.gz' 'full_run.log' '*Token.txt' > "$CLONE/$REL/.gitignore"

  PUBLISHED_TERMINAL="$CLONE/$REL/Terminal.txt"
  if [ -f "$PUBLISHED_TERMINAL" ] &&
     [ "$(stat -c %s "$PUBLISHED_TERMINAL")" -gt $((8 * 1024 * 1024)) ]; then
    echo "Compacting oversized Terminal.txt before GitHub publication..."
    "$BASE/Codes/envs/sharp/bin/python" - \
      "$PUBLISHED_TERMINAL" $((8 * 1024 * 1024)) <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
limit = int(sys.argv[2])
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
marker = "[middle of terminal transcript omitted; complete log is retained on Lab 2]"
head = []
used = len((marker + "\n").encode("utf-8"))
head_budget = limit // 3
for line in lines:
    encoded = len((line + "\n").encode("utf-8"))
    if used + encoded > head_budget:
        break
    head.append(line)
    used += encoded
tail = []
for line in reversed(lines[len(head):]):
    encoded = len((line + "\n").encode("utf-8"))
    if used + encoded > limit:
        break
    tail.append(line)
    used += encoded
path.write_text(
    "\n".join(head + [marker] + list(reversed(tail))) + "\n",
    encoding="utf-8",
)
PY
  fi

  LARGE=$(find "$CLONE/$REL" "$CLONE/$COMPARISON_REL" \
    -type f -size +10M -print 2>/dev/null)
  CREDENTIALS=$(grep -RIlE 'github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+' \
    "$CLONE/$REL" "$CLONE/$COMPARISON_REL" 2>/dev/null)
  if [ -n "$LARGE" ]; then
    echo "ERROR: publication contains files larger than 10 MiB:"
    echo "$LARGE"
    STATUS=1
  elif [ -n "$CREDENTIALS" ]; then
    echo "ERROR: credential text was detected in publication artifacts."
    STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  "$GIT" -C "$CLONE" config user.name madviddd
  "$GIT" -C "$CLONE" config user.email madviddd@users.noreply.github.com
  "$GIT" -C "$CLONE" config pull.rebase false
  "$GIT" -C "$CLONE" config merge.autoStash true
  "$GIT" -C "$CLONE" add -- "$REL" "$COMPARISON_REL"
  if "$GIT" -C "$CLONE" diff --cached --quiet; then
    echo "No new GitHub artifact changes required a commit."
  else
    "$GIT" -C "$CLONE" commit -m "Publish final SHARP suite: $SLUG"
    STATUS=$?
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  STATUS=1
  for ATTEMPT in 1 2 3; do
    GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
      "$GIT" -C "$CLONE" -c credential.helper= \
      pull --no-rebase origin main
    PULL_STATUS=$?
    if [ "$PULL_STATUS" -ne 0 ]; then
      echo "ERROR: automatic merge from remote main failed."
      STATUS="$PULL_STATUS"
      break
    fi

    GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
      "$GIT" -C "$CLONE" -c credential.helper= push origin main
    STATUS=$?
    [ "$STATUS" -eq 0 ] && break
    echo "Push attempt $ATTEMPT failed; automatically merging remote main and retrying."
    sleep 10
  done
fi

if [ "$STATUS" -eq 0 ]; then
  COMMIT=$("$GIT" -C "$CLONE" rev-parse HEAD)
  printf 'published_commit=%s\npublished_path=%s\n' "$COMMIT" "$REL" \
    > "$RESULTS_ROOT/$SLUG/PUBLISHED"
  echo "FINAL_RUN_ARTIFACTS_PUBLISHED"
  echo "Published commit: $COMMIT"
  echo "Published path: $REL"
else
  echo "ERROR: automatic result publication failed."
fi

rm -f "$ASKPASS"
unset FINAL3_GITHUB_TOKEN
exit "$STATUS"
