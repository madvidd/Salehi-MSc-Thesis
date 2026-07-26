# Clean restart of the remaining Lab 3 attention suite

The supplied log contains no training exception. It contains three compatibility
warning groups from third-party compatibility plus Lightning batch-size
inference. The update removes the deprecated import, filters only verified
compatibility notices, explicitly logs the real per-rank validation batch size,
validates every custom attention variant with a forward/backward smoke test, and
keeps the model and all training parameters unchanged.

Run this entire block in a new Lab 3 terminal. It does not close any terminal,
does not delete previous results, explicitly excludes baseline MHA, preserves the
partial qknorm attempt, updates the code using the PAT in `M/Token/Token.txt`, and
restarts qknorm, talking-heads, then qknorm+talking-heads in the foreground.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

restart_clean_attention_suite() {
  BASE=/home/server01/M
  TOKEN_FILE="$BASE/Token/Token.txt"
  REPO=$(tr -d '\r\n' < "$BASE/Codes/LATEST_THESIS_LAB3_CLONE.txt")
  ROOT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")
  RESULTS=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
  ENV="$BASE/Codes/AV2/envs/sharp_av2"
  WORK_BRANCH=lab3-sharp-attention-ablation

  echo "Stopping only the existing remaining-attention suite..."
  SUITE_PID=$(pgrep -u "$USER" -fo '[r]un_remaining_attention_suite.sh')
  if [ -n "$SUITE_PID" ]; then
    PGID=$(ps -o pgid= -p "$SUITE_PID" | tr -d ' ')
    SHELL_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
    if [ -n "$PGID" ] && [ "$PGID" != "$SHELL_PGID" ]; then
      kill -INT -- "-$PGID" 2>/dev/null
    else
      kill -INT "$SUITE_PID" 2>/dev/null
    fi
  fi

  for _ in $(seq 1 30); do
    pgrep -u "$USER" -f '[r]un_remaining_attention_suite.sh|[t]rain.py.*SHARP_ATTENTION_ABLATION' >/dev/null || break
    sleep 1
  done

  REMAINING=$(pgrep -u "$USER" -f '[r]un_remaining_attention_suite.sh|[t]rain.py.*SHARP_ATTENTION_ABLATION')
  if [ -n "$REMAINING" ]; then
    echo "Stopping remaining suite processes: $REMAINING"
    kill -TERM $REMAINING 2>/dev/null
    sleep 10
  fi

  if pgrep -u "$USER" -f '[r]un_remaining_attention_suite.sh|[t]rain.py.*SHARP_ATTENTION_ABLATION' >/dev/null; then
    echo "ERROR: the previous suite is still active; clean restart cancelled."
    return 1
  fi

  if [ -d "$RESULTS/qknorm" ] && [ ! -f "$RESULTS/qknorm/COMPLETE" ]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    PRESERVED="$RESULTS/preserved_partial_runs/qknorm_before_clean_$STAMP"
    mkdir -p "$(dirname "$PRESERVED")"
    mv "$RESULTS/qknorm" "$PRESERVED"
    echo "Partial qknorm run preserved at: $PRESERVED"
  fi

  [ -f "$TOKEN_FILE" ] || { echo "ERROR: missing $TOKEN_FILE"; return 1; }
  [ -d "$REPO/.git" ] || { echo "ERROR: missing repository $REPO"; return 1; }
  [ -x "$ENV/bin/python" ] || { echo "ERROR: missing environment $ENV"; return 1; }
  chmod 600 "$TOKEN_FILE"

  TOKEN=$("$ENV/bin/python" -c '
import pathlib, re, sys
raw = pathlib.Path(sys.argv[1]).read_bytes()
text = raw.decode("utf-8-sig", "ignore") + "\n" + raw.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
' "$TOKEN_FILE")
  [ -n "$TOKEN" ] || { echo "ERROR: no PAT found in Token.txt"; return 1; }

  USER_JSON=$(mktemp) || return 1
  REPO_JSON=$(mktemp) || { rm -f "$USER_JSON"; return 1; }
  USER_HTTP=$(curl -sS -o "$USER_JSON" -w '%{http_code}' \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user)
  REPO_HTTP=$(curl -sS -o "$REPO_JSON" -w '%{http_code}' \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/madviddd/Thesis)
  LOGIN=$("$ENV/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("login", ""))' \
    "$USER_JSON" 2>/dev/null)
  ACCESS=$("$ENV/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("full_name", ""))' \
    "$REPO_JSON" 2>/dev/null)
  PUSH=$("$ENV/bin/python" -c \
    'import json,sys; print(str(json.load(open(sys.argv[1])).get("permissions", {}).get("push", False)).lower())' \
    "$REPO_JSON" 2>/dev/null)
  rm -f "$USER_JSON" "$REPO_JSON"

  if [ "$USER_HTTP" != 200 ] || [ "$REPO_HTTP" != 200 ] || \
     [ "$LOGIN" != madvidd ] || [ "$ACCESS" != madviddd/Thesis ] || \
     [ "$PUSH" != true ]; then
    echo "ERROR: PAT verification failed: user_http=$USER_HTTP repo_http=$REPO_HTTP account=$LOGIN repository=$ACCESS push=$PUSH"
    unset TOKEN
    return 1
  fi
  echo "PAT verified for $LOGIN with push access to $ACCESS."

  export LAB3_GITHUB_TOKEN="$TOKEN"
  unset TOKEN
  ASKPASS=$(mktemp) || return 1
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    '  *Username*) printf "%s\n" "madvidd" ;;' \
    '  *Password*) printf "%s\n" "$LAB3_GITHUB_TOKEN" ;;' \
    'esac' > "$ASKPASS"
  chmod 700 "$ASKPASS"

  cd "$REPO" || return 1
  git remote set-url origin https://github.com/madviddd/Thesis.git
  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= fetch origin "$WORK_BRANCH" main || {
      rm -f "$ASKPASS"
      unset LAB3_GITHUB_TOKEN
      return 1
    }

  git switch "$WORK_BRANCH" 2>/dev/null || \
    git switch --track -c "$WORK_BRANCH" "origin/$WORK_BRANCH" || {
      rm -f "$ASKPASS"
      unset LAB3_GITHUB_TOKEN
      return 1
    }

  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= pull --ff-only origin "$WORK_BRANCH"
  STATUS=$?
  rm -f "$ASKPASS"
  unset LAB3_GITHUB_TOKEN

  [ "$STATUS" -eq 0 ] || return "$STATUS"

  chmod +x "Lab 3/Codes/run_remaining_attention_suite.sh"
  "$ENV/bin/python" "Lab 3/Codes/prepare_remaining_attention_runtime.py" || return 1

  echo "Starting clean remaining-attention suite in this terminal."
  echo "Baseline MHA will not run. Previous results remain preserved."
  bash "Lab 3/Codes/run_remaining_attention_suite.sh"
  STATUS=$?

  echo "Final suite status: $STATUS"
  echo "Terminal remains open."
  return "$STATUS"
}

restart_clean_attention_suite
STATUS=$?
unset -f restart_clean_attention_suite

echo
echo "Terminal remains open."
echo "Final command status: $STATUS"
```