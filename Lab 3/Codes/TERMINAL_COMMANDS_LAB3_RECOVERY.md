# Lab 3 hardened non-baseline attention restart

This procedure preserves the existing baseline and partial attention results,
uses token-only system Git as `madviddd`, validates PyTorch 2.8 with CUDA 12.6,
stress-tests all three GPUs, and runs these variants from epoch 0 in sequence:

1. `qknorm`
2. `talking_heads`
3. `qknorm_talking_heads`

Each variant has four checkpoint-aware attempts. After each variant, its complete
local archive is created. Only files smaller than 10 MB are committed directly
to `main`; pull, merge, push, and remote-SHA verification must succeed before
the next variant starts. Baseline MHA is never restarted.

Run this block in a separate Lab 3 terminal. The training suite itself then runs
in this same terminal. No `sudo`, browser, GitHub CLI, `tmux`, or terminal exit is
used.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

start_hardened_lab3_suite() {
  BASE=/home/server01/M
  TOKEN_FILE="$BASE/Token/Token.txt"
  GIT=/usr/bin/git
  STAMP=$(date +%Y%m%d-%H%M%S)
  CONTROL="$BASE/Codes/Thesis_Lab3_Hardened_$STAMP"
  ASKPASS="$BASE/Token/git-token-askpass.sh"

  [ -x "$GIT" ] || {
    echo "ERROR: /usr/bin/git is unavailable."
    return 1
  }
  [ -s "$TOKEN_FILE" ] || {
    echo "ERROR: missing $TOKEN_FILE"
    return 1
  }

  TOKEN=$(python3 - "$TOKEN_FILE" <<'PY'
import pathlib
import re
import sys

data = pathlib.Path(sys.argv[1]).read_bytes()
match = re.search(rb"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", data)
if match:
    print(match.group(0).decode())
else:
    text = data.decode("utf-16", errors="ignore")
    match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
    print(match.group(0) if match else "")
PY
)
  [ -n "$TOKEN" ] || {
    echo "ERROR: no PAT found in Token.txt"
    return 1
  }

  USER_JSON=$(mktemp)
  REPO_JSON=$(mktemp)
  USER_HTTP=$(curl -sS -o "$USER_JSON" -w '%{http_code}' \
    -H "Authorization: Bearer $TOKEN" \
    https://api.github.com/user)
  REPO_HTTP=$(curl -sS -o "$REPO_JSON" -w '%{http_code}' \
    -H "Authorization: Bearer $TOKEN" \
    https://api.github.com/repos/madvidd/Thesis)
  LOGIN=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("login",""))' \
    "$USER_JSON" 2>/dev/null)
  PUSH=$(python3 -c \
    'import json,sys; print(str(json.load(open(sys.argv[1])).get("permissions",{}).get("push",False)).lower())' \
    "$REPO_JSON" 2>/dev/null)
  rm -f "$USER_JSON" "$REPO_JSON"

  echo "Token account: $LOGIN"
  echo "User API: $USER_HTTP; repository API: $REPO_HTTP; push: $PUSH"
  if [ "$USER_HTTP" != 200 ] || [ "$REPO_HTTP" != 200 ] || \
     [ "$LOGIN" != madviddd ] || [ "$PUSH" != true ]; then
    echo "ERROR: Token.txt is not a writable madviddd token."
    unset TOKEN
    return 1
  fi

  printf '%s\n' "$TOKEN" > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
  unset TOKEN

  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    '  *Username*) printf "%s\n" "madviddd" ;;' \
    '  *Password*) tr -d "\r\n[:space:]" < /home/server01/M/Token/Token.txt ;;' \
    'esac' > "$ASKPASS"
  chmod 700 "$ASKPASS"

  unset GIT_TEMPLATE_DIR GIT_EXEC_PATH GH_TOKEN GITHUB_TOKEN
  export GIT_EXEC_PATH=$("$GIT" --exec-path)
  "$GIT" credential-cache exit 2>/dev/null || true

  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    "$GIT" -c credential.helper= clone \
      --branch main --single-branch \
      https://github.com/madvidd/Thesis.git "$CONTROL" || return 1

  "$GIT" -C "$CONTROL" config credential.helper ""
  "$GIT" -C "$CONTROL" config core.askPass "$ASKPASS"
  "$GIT" -C "$CONTROL" config credential.username madviddd
  "$GIT" -C "$CONTROL" config pull.rebase false
  "$GIT" -C "$CONTROL" config merge.autoStash true

  printf '%s\n' "$CONTROL" \
    > "$BASE/Codes/LATEST_THESIS_LAB3_CLONE.txt"

  echo "HARDENED_CONTROL_CLONE=$CONTROL"
  bash "$CONTROL/Lab 3/Codes/recover_and_restart_lab3.sh"
  return $?
}

start_hardened_lab3_suite
STATUS=$?
unset -f start_hardened_lab3_suite

echo
echo "Terminal remains open."
echo "Final command status: $STATUS"
```