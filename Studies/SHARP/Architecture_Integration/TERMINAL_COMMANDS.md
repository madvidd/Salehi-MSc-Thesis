# Lab 2 Terminal Command

Paste the following complete block into a new Lab 2 terminal. It reads the `madviddd` token from `/home/server00/M/Token/Token.txt`, updates and automatically merges `main`, then starts or resumes the foreground suite. It never calls `exit` in the interactive parent shell.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server00/M
REPO="$BASE/Codes/Thesis"
TOKEN_FILE="$BASE/Token/Token.txt"
GIT=/usr/bin/git
STATUS=1

TOKEN=$("$BASE/Codes/envs/sharp/bin/python" - "$TOKEN_FILE" <<'PY'
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
  "$BASE/Codes/envs/sharp/bin/python" -c \
  'import json,sys; print(json.load(sys.stdin).get("login", ""))' 2>/dev/null)

echo "GitHub account: $LOGIN"
if [ "$LOGIN" = "madviddd" ] && [ -d "$REPO/.git" ]; then
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
  unset FINAL3_GITHUB_TOKEN TOKEN
else
  echo "ERROR: Token.txt is not a valid madviddd token or the Thesis clone is missing."
fi

PACKAGE="$REPO/Studies/SHARP/Architecture_Integration"
if [ "$STATUS" -eq 0 ]; then
  chmod 700 "$PACKAGE"/*.sh "$PACKAGE"/*.py
  cd "$PACKAGE"
  bash launch_final_3run_suite.sh
  STATUS=$?
fi

echo
echo "Final suite command status: $STATUS"
echo "The terminal remains open."
```

If the machine, network, or terminal interrupts training, paste the same block again. It reuses the same result root and resumes the interrupted variant from the newest valid checkpoint.

## Resume after an oversized terminal blocked publication

The package also includes `resume_after_terminal_publication_fix.sh`. It refuses to skip Run 1 unless the completion marker, final metrics, and best checkpoint all validate. It then stores the full suite log and Run 1 evidence under `/home/server00/M/Terminal/SHARP_Final_3_Run_Suite`, refreshes the corrected orchestration scripts, republishes a bounded Run 1 transcript, and continues Runs 2 and 3 without retraining Run 1.
