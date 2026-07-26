#!/usr/bin/env bash

# User-space recovery for the remaining Lab 3 attention sequence.
# This script never uses sudo and is intended to be run, not sourced.

set -uo pipefail

BASE=/home/server01/M
TOKEN_FILE="$BASE/Token/Token.txt"
CONDA_PYTHON="$BASE/Codes/AV2/miniforge3/bin/python"
WORK_BRANCH=lab3-sharp-attention-ablation
REPO=$(tr -d '\r\n' < "$BASE/Codes/LATEST_THESIS_LAB3_CLONE.txt")
ROOT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")
RESULTS=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_ROOT="$RESULTS/preserved_partial_runs/restart_$STAMP"

fail() {
  echo "FATAL: $*" >&2
  echo "The SSH shell remains open after this script returns."
  exit 1
}

for required in \
  "$TOKEN_FILE" \
  "$CONDA_PYTHON" \
  "$REPO/.git" \
  "$ROOT" \
  "$RESULTS" \
  "$SCRIPT_DIR/prepare_lab3_cuda126_env.sh"
do
  [[ -e "$required" ]] || fail "required path is missing: $required"
done

command -v gh >/dev/null 2>&1 || fail "gh is not installed"
command -v tmux >/dev/null 2>&1 || fail "tmux is not installed"

chmod 600 "$TOKEN_FILE"
TOKEN=$("$CONDA_PYTHON" -c '
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
' "$TOKEN_FILE")
[[ -n "$TOKEN" ]] || fail "no GitHub PAT was found in $TOKEN_FILE"

echo "Removing GitHub CLI authentication for madvidd..."
gh auth logout --hostname github.com --user madvidd >/dev/null 2>&1 || true
printf 'protocol=https\nhost=github.com\nusername=madvidd\n\n' \
  | git credential reject >/dev/null 2>&1 || true

echo "Replacing the active GitHub authentication with Token.txt..."
gh auth logout --hostname github.com --user madviddd >/dev/null 2>&1 || true
printf '%s\n' "$TOKEN" | gh auth login --hostname github.com --with-token \
  || fail "the replacement token was rejected"
gh auth switch --hostname github.com --user madviddd \
  || fail "madviddd could not be selected"
gh auth setup-git || fail "gh could not configure Git credentials"

LOGIN=$(gh api user --jq .login 2>/dev/null)
[[ "$LOGIN" == madviddd ]] || fail "active GitHub account is $LOGIN, expected madviddd"
echo "GITHUB_AUTHENTICATED_ACCOUNT=$LOGIN"

git config --global user.name "Seyed Mohammad Salehi"
git config --global user.email "madviddd@users.noreply.github.com"

echo "Running a real pull/integrate/push validation against Trading_Bot..."
AUTH_TEST_REPO="$BASE/Git/Trading_Bot_auth_test"
mkdir -p "$(dirname "$AUTH_TEST_REPO")"
if [[ ! -d "$AUTH_TEST_REPO/.git" ]]; then
  gh repo clone madvidd/Trading_Bot "$AUTH_TEST_REPO" \
    || fail "Trading_Bot clone failed"
fi
git -C "$AUTH_TEST_REPO" remote set-url origin \
  https://github.com/madvidd/Trading_Bot.git
git -C "$AUTH_TEST_REPO" switch main \
  || fail "Trading_Bot main switch failed"
git -C "$AUTH_TEST_REPO" pull --rebase origin main \
  || fail "Trading_Bot pull/rebase failed"
git -C "$AUTH_TEST_REPO" fetch origin main \
  || fail "Trading_Bot fetch failed"
git -C "$AUTH_TEST_REPO" merge --ff-only origin/main \
  || fail "Trading_Bot integration failed"
TEST_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git -C "$AUTH_TEST_REPO" -c commit.gpgsign=false commit --allow-empty \
  -m "Verify madviddd Lab 3 Git synchronization $TEST_TIME" \
  || fail "Trading_Bot validation commit failed"
TEST_SHA=$(git -C "$AUTH_TEST_REPO" rev-parse HEAD)
git -C "$AUTH_TEST_REPO" pull --rebase origin main \
  || fail "Trading_Bot final pull/rebase failed"
git -C "$AUTH_TEST_REPO" push origin HEAD:main \
  || fail "Trading_Bot push failed"
REMOTE_SHA=$(gh api "repos/madvidd/Trading_Bot/commits/$TEST_SHA" --jq .sha 2>/dev/null)
[[ "$REMOTE_SHA" == "$TEST_SHA" ]] \
  || fail "GitHub did not confirm the Trading_Bot validation commit"
echo "TRADING_BOT_PULL_MERGE_PUSH_OK=$TEST_SHA"

echo "Stopping only the existing remaining-attention suite..."
mapfile -t SUITE_PIDS < <(
  pgrep -u "$USER" -f \
    '[r]un_remaining_attention_suite.sh|[r]un_remaining_attention_suite_resilient.sh'
)
CURRENT_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
for pid in "${SUITE_PIDS[@]}"; do
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [[ -n "$pgid" && "$pgid" != "$CURRENT_PGID" ]]; then
    kill -INT -- "-$pgid" 2>/dev/null || true
  else
    kill -INT "$pid" 2>/dev/null || true
  fi
done

for _ in $(seq 1 60); do
  pgrep -u "$USER" -f \
    '[r]un_remaining_attention_suite.sh|[r]un_remaining_attention_suite_resilient.sh' \
    >/dev/null || break
  sleep 1
done

if pgrep -u "$USER" -f \
  '[r]un_remaining_attention_suite.sh|[r]un_remaining_attention_suite_resilient.sh' \
  >/dev/null
then
  fail "the previous suite did not stop; no files or results were moved"
fi

mkdir -p "$BACKUP_ROOT"
for variant in qknorm talking_heads qknorm_talking_heads; do
  if [[ -d "$RESULTS/$variant" ]]; then
    mv "$RESULTS/$variant" "$BACKUP_ROOT/$variant"
    echo "PRESERVED_PREVIOUS_RUN[$variant]=$BACKUP_ROOT/$variant"
  fi
done

echo "Updating the Lab 3 source branch..."
if ! git -C "$REPO" diff --quiet -- "Lab 3/Codes/run_remaining_attention_suite.sh"; then
  git -C "$REPO" diff -- "Lab 3/Codes/run_remaining_attention_suite.sh" \
    > "$BACKUP_ROOT/previous_launcher.patch"
  git -C "$REPO" restore --source=HEAD -- \
    "Lab 3/Codes/run_remaining_attention_suite.sh"
  echo "PRESERVED_PREVIOUS_LAUNCHER=$BACKUP_ROOT/previous_launcher.patch"
fi

DIRTY=$(git -C "$REPO" status --porcelain)
[[ -z "$DIRTY" ]] || {
  printf '%s\n' "$DIRTY"
  fail "the Thesis clone has other uncommitted changes; update was not attempted"
}

git -C "$REPO" remote set-url origin https://github.com/madviddd/Thesis.git
git -C "$REPO" switch "$WORK_BRANCH" \
  || fail "could not switch to $WORK_BRANCH"
git -C "$REPO" fetch --prune origin \
  || fail "Thesis fetch failed"
git -C "$REPO" pull --rebase origin "$WORK_BRANCH" \
  || fail "Thesis pull/rebase failed"
echo "THESIS_BRANCH_UPDATED=$(git -C "$REPO" rev-parse HEAD)"

echo "Preparing and validating the user-owned CUDA 12.6 environment..."
bash "$REPO/Lab 3/Codes/prepare_lab3_cuda126_env.sh" \
  || fail "CUDA 12.6 environment preparation or three-GPU stress testing failed"

SESSION=lab3_attention
if tmux has-session -t "$SESSION" 2>/dev/null; then
  SESSION="${SESSION}_$STAMP"
fi
LAUNCHER="$REPO/Lab 3/Codes/run_remaining_attention_suite.sh"
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "bash '$LAUNCHER'; rc=\$?; echo; echo LAB3_SUITE_EXIT_CODE=\$rc; echo 'The tmux pane remains open.'; exec bash"
tmux set-option -t "$SESSION" remain-on-exit on

echo "LAB3_RESTARTED_IN_TMUX=$SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Baseline MHA remains excluded."
echo "The SSH shell remains open."
unset TOKEN
