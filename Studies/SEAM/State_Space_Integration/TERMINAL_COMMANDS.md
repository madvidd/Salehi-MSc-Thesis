# Lab 3 Terminal Commands

Paste the following block into a new Lab 3 terminal. It authenticates Git using
the `madviddd` PAT in `/home/server01/M/Token/Token.txt`, creates a clean clone of
`main`, and runs the three experiments in the foreground.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server01/M
TOKEN_FILE="$BASE/Token/Token.txt"
GIT=/usr/bin/git
STAMP=$(date +%Y%m%d-%H%M%S)
REPO="$BASE/Codes/Thesis_SEAM_Launch_$STAMP"

TOKEN=$(python3 - "$TOKEN_FILE" <<'PY'
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
PY
)

LOGIN=$(curl -fsSL -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/user 2>/dev/null | \
  python3 -c 'import json,sys; print(json.load(sys.stdin).get("login",""))' \
  2>/dev/null)

STATUS=1
if [ "$LOGIN" = "madviddd" ]; then
  export LAB3_GITHUB_TOKEN="$TOKEN"
  ASKPASS=$(mktemp)
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    ' *Username*) printf "%s\n" "madviddd" ;;' \
    ' *Password*) printf "%s\n" "$LAB3_GITHUB_TOKEN" ;;' \
    'esac' > "$ASKPASS"
  chmod 700 "$ASKPASS"

  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    "$GIT" -c credential.helper= clone \
    --depth 1 --single-branch --branch main \
    https://github.com/madvidd/Thesis.git "$REPO"
  STATUS=$?

  rm -f "$ASKPASS"
  unset LAB3_GITHUB_TOKEN TOKEN

  if [ "$STATUS" -eq 0 ]; then
    PACKAGE="$REPO/Studies/SEAM/State_Space_Integration"
    cd "$PACKAGE"
    bash "$PACKAGE/launch_lab3_seam_3run.sh"
    STATUS=$?
  fi
else
  echo "ERROR: Token.txt is not a madviddd PAT."
fi

echo
echo "SEAM suite status: $STATUS"
echo "Terminal remains open."
```
