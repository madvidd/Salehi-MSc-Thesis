#!/usr/bin/env bash

# User-space recovery for all non-baseline Lab 3 attention variants. This script
# uses no sudo, no GitHub CLI, and no background terminal manager.

set -uo pipefail

BASE=/home/server01/M
TOKEN_FILE="$BASE/Token/Token.txt"
CONDA_PYTHON="$BASE/Codes/AV2/miniforge3/bin/python"
ROOT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")
RESULTS=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONTROL_REPO=$(/usr/bin/git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
STAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_ROOT="$RESULTS/preserved_partial_runs/full_restart_$STAMP"
GIT=/usr/bin/git
ASKPASS="$BASE/Token/git-token-askpass.sh"

fail() {
  echo "FATAL: $*" >&2
  echo "The terminal remains open after this script returns."
  exit 1
}

for required in \
  "$TOKEN_FILE" \
  "$CONDA_PYTHON" \
  "$CONTROL_REPO/.git" \
  "$ROOT" \
  "$RESULTS" \
  "$SCRIPT_DIR/prepare_lab3_cuda126_env.sh" \
  "$SCRIPT_DIR/run_remaining_attention_suite.sh"
do
  [[ -e "$required" ]] || fail "required path is missing: $required"
done

chmod 600 "$TOKEN_FILE"
TOKEN=$("$CONDA_PYTHON" - "$TOKEN_FILE" <<'PY'
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
[[ -n "$TOKEN" ]] || fail "no GitHub PAT was found in $TOKEN_FILE"

USER_JSON=$(mktemp)
REPO_JSON=$(mktemp)
USER_HTTP=$(curl -sS -o "$USER_JSON" -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/user)
REPO_HTTP=$(curl -sS -o "$REPO_JSON" -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  https://api.github.com/repos/madvidd/Thesis)
LOGIN=$("$CONDA_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1])).get("login",""))' \
  "$USER_JSON" 2>/dev/null)
PUSH=$("$CONDA_PYTHON" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1])).get("permissions",{}).get("push",False)).lower())' \
  "$REPO_JSON" 2>/dev/null)
rm -f "$USER_JSON" "$REPO_JSON"

if [[ "$USER_HTTP" != 200 || "$REPO_HTTP" != 200 || \
      "$LOGIN" != madviddd || "$PUSH" != true ]]; then
  unset TOKEN
  fail "Token.txt is not a writable madviddd token"
fi
echo "GITHUB_AUTHENTICATED_ACCOUNT=$LOGIN"

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

"$GIT" -C "$CONTROL_REPO" remote set-url origin \
  https://github.com/madvidd/Thesis.git
"$GIT" -C "$CONTROL_REPO" config credential.helper ""
"$GIT" -C "$CONTROL_REPO" config core.askPass "$ASKPASS"
"$GIT" -C "$CONTROL_REPO" config credential.username madviddd
"$GIT" -C "$CONTROL_REPO" config pull.rebase false
"$GIT" -C "$CONTROL_REPO" config merge.autoStash true

GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
  "$GIT" -C "$CONTROL_REPO" -c credential.helper= \
  ls-remote origin HEAD >/dev/null \
  || fail "token-only Git access test failed"
echo "TOKEN_ONLY_GIT_ACCESS_OK=True"

echo "Stopping only the active non-baseline attention suite..."
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
  for pid in "${SUITE_PIDS[@]}"; do
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ -n "$pgid" && "$pgid" != "$CURRENT_PGID" ]]; then
      kill -TERM -- "-$pgid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 15
fi

if pgrep -u "$USER" -f \
  '[r]un_remaining_attention_suite.sh|[r]un_remaining_attention_suite_resilient.sh' \
  >/dev/null
then
  fail "the previous suite did not stop; no result directories were moved"
fi

while IFS= read -r pid; do
  [[ -n "$pid" ]] || continue
  cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
  if [[ "$cwd" == "$ROOT/variants/"* ]]; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
done < <(pgrep -u "$USER" -f '[p]ython.*train.py' || true)
sleep 10

mkdir -p "$BACKUP_ROOT"
for variant in qknorm talking_heads qknorm_talking_heads; do
  if [[ -d "$RESULTS/$variant" ]]; then
    mv "$RESULTS/$variant" "$BACKUP_ROOT/$variant"
    echo "PRESERVED_PREVIOUS_RUN[$variant]=$BACKUP_ROOT/$variant"
  fi
done
echo "BASELINE_MHA_PRESERVED=$RESULTS/baseline_mha"

echo "Preparing PyTorch 2.8 CUDA 12.6 and stress-testing all variants..."
LAB3_CUDA_STRESS_ITERATIONS=1000 \
  bash "$SCRIPT_DIR/prepare_lab3_cuda126_env.sh" \
  || fail "CUDA 12.6 preparation or three-GPU stress test failed"

ENV=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ENV.txt")
"$ENV/bin/python" - <<'PY' || fail "final CUDA runtime verification failed"
import torch

assert torch.__version__.startswith("2.8.0"), torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 3
print(f"FINAL_RUNTIME_OK={torch.__version__} cuda={torch.version.cuda}")
PY

echo "Starting qknorm, talking_heads, and qknorm_talking_heads in this terminal."
echo "Each run is archived and merged into GitHub main before the next begins."
echo "Baseline MHA remains excluded."
bash "$SCRIPT_DIR/run_remaining_attention_suite.sh"
STATUS=$?

echo
echo "LAB3_SUITE_EXIT_CODE=$STATUS"
echo "Previous partial runs remain at: $BACKUP_ROOT"
echo "The terminal remains open."
exit "$STATUS"