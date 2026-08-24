# Lab 3 Terminal Commands

Paste the following block into a new Lab 3 terminal. It authenticates Git using
the `madviddd` PAT in `/home/server01/M/Token/Token.txt`, updates `main`, and runs
the three experiments in the foreground.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server01/M
REPO=$(tr -d '\r\n' < "$BASE/Codes/LATEST_THESIS_LAB3_CLONE.txt" 2>/dev/null)
TOKEN_FILE="$BASE/Token/Token.txt"
GIT=/usr/bin/git

[ -d "$REPO/.git" ] || REPO="$BASE/Codes/Thesis"

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
if [ "$LOGIN" = "madviddd" ] && [ -d "$REPO/.git" ]; then
  export LAB3_GITHUB_TOKEN="$TOKEN"
  ASKPASS=$(mktemp)
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    ' *Username*) printf "%s\n" "madviddd" ;;' \
    ' *Password*) printf "%s\n" "$LAB3_GITHUB_TOKEN" ;;' \
    'esac' > "$ASKPASS"
  chmod 700 "$ASKPASS"

  cd "$REPO"
  "$GIT" remote set-url origin https://github.com/madvidd/Thesis.git
  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    "$GIT" -c credential.helper= fetch origin main
  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    "$GIT" switch main
    "$GIT" merge --autostash FETCH_HEAD
    STATUS=$?
  fi

  rm -f "$ASKPASS"
  unset LAB3_GITHUB_TOKEN TOKEN

  if [ "$STATUS" -eq 0 ]; then
    PACKAGE="$REPO/Lab 3/SEAM_AV2_Mamba_3_Run"
    cd "$PACKAGE"
    bash "$PACKAGE/launch_lab3_seam_3run.sh"
    STATUS=$?
  fi
else
  echo "ERROR: Token.txt is not a madviddd PAT or the Thesis clone is missing."
fi

echo
echo "SEAM suite status: $STATUS"
echo "Terminal remains open."
```
