# Lab 3 Max-Resource Launch Commands

Run this block in a new Lab 3 terminal. It preserves and stops only the superseded active batch-32 SEAM suite, downloads the current package using the `madviddd` token in `Token.txt`, and starts the independent three-GPU suite. It does not delete previous code, logs, events, checkpoints, or results.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server01/M
TOKEN_FILE="$BASE/Token/Token.txt"
GIT=/usr/bin/git
STAMP=$(date +%Y%m%d-%H%M%S)
CLONE="$BASE/Codes/Thesis_SEAM20_MaxResources_$STAMP"
STATUS=1

TOKEN=$(python3 - "$TOKEN_FILE" <<'PY'
from pathlib import Path
import re
import sys

data = Path(sys.argv[1]).read_bytes()
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

if [ "$LOGIN" = "madviddd" ] && [ "$PUSH" = "true" ]; then
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

  rm -f "$ASKPASS"
  unset TOKEN LAB3_GITHUB_TOKEN

  if [ "$STATUS" -eq 0 ]; then
    PACKAGE="$CLONE/Lab 3/SEAM_20_Epoch_4_Test_Suite"

    bash "$PACKAGE/stop_current_seam_20epoch_suite.sh"
    STATUS=$?

    if [ "$STATUS" -eq 0 ]; then
      bash "$PACKAGE/launch_lab3_seam_20epoch_4test_max_resources.sh"
      STATUS=$?
    fi
  fi
else
  echo "ERROR: Token.txt is not a writable madviddd token."
  unset TOKEN
fi

echo
echo "Max-resource suite status: $STATUS"
echo "Terminal remains open."
```
