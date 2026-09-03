# Lab 3 Launch or Resume Command

Paste the complete block below into one new Lab 3 terminal. It authenticates non-interactively with the writable `madviddd` token in `/home/server01/M/Token/Token.txt`, downloads the current `main` branch into a new isolated clone, and launches or resumes the four-run suite. It does not stop, modify, or delete any previous experiment.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server01/M
TOKEN_FILE="$BASE/Token/Token.txt"
GIT=/usr/bin/git
STAMP=$(date +%Y%m%d-%H%M%S)
CLONE="$BASE/Codes/Thesis_SEAM20_Launch_$STAMP"
STATUS=1

TOKEN=$(python3 - "$TOKEN_FILE" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
values = []
if path.is_file():
    data = path.read_bytes()
    values.extend((
        data.decode("utf-8-sig", "ignore"),
        data.decode("utf-16", "ignore"),
    ))

for value in values:
    for token in re.findall(
        r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", value
    ):
        print(token)
PY
)

SELECTED=""
while IFS= read -r CANDIDATE; do
  [ -n "$CANDIDATE" ] || continue
  LOGIN=$(curl -fsSL -H "Authorization: Bearer $CANDIDATE" \
    https://api.github.com/user 2>/dev/null | \
    python3 -c 'import json,sys; print(json.load(sys.stdin).get("login",""))' \
    2>/dev/null)
  PUSH=$(curl -fsSL -H "Authorization: Bearer $CANDIDATE" \
    https://api.github.com/repos/madvidd/Thesis 2>/dev/null | \
    python3 -c 'import json,sys; print(str(json.load(sys.stdin).get("permissions",{}).get("push",False)).lower())' \
    2>/dev/null)
  if [ "$LOGIN" = "madviddd" ] && [ "$PUSH" = "true" ]; then
    SELECTED="$CANDIDATE"
    break
  fi
done <<< "$TOKEN"
unset TOKEN CANDIDATE

if [ -z "$SELECTED" ]; then
  echo "ERROR: Token.txt contains no writable madviddd PAT."
else
  export LAB3_GITHUB_TOKEN="$SELECTED"
  unset SELECTED
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

  rm -f "$ASKPASS"
  unset LAB3_GITHUB_TOKEN

  if [ "$STATUS" -eq 0 ]; then
    PACKAGE="$CLONE/Lab 3/SEAM_20_Epoch_4_Test_Suite"
    cd "$PACKAGE" || STATUS=1
  fi

  if [ "$STATUS" -eq 0 ]; then
    bash "$PACKAGE/launch_lab3_seam_20epoch_4test.sh"
    STATUS=$?
  fi
fi

echo
echo "SEAM four-test suite status: $STATUS"
echo "The same block safely resumes from the latest per-epoch checkpoint."
echo "Terminal remains open."
```
