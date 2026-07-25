# Lab 2 terminal commands

## 1. Pull the updated package from `main` with the active madviddd token

```bash
BASE=/home/server00/M
REPO="$BASE/Codes/Thesis"
BRANCH=main

cd "$REPO"
gh auth switch --hostname github.com --user madviddd
test "$(gh api user --jq .login)" = "madviddd"

ASKPASS="$REPO/.git/gh-token-askpass.sh"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'case "$1" in' \
  '  *Username*) printf "%s\n" "madviddd" ;;' \
  '  *Password*) gh auth token --hostname github.com ;;' \
  'esac' > "$ASKPASS"
chmod 700 "$ASKPASS"

git config --local core.askPass "$ASKPASS"
git config --local credential.username madviddd

GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
  git fetch origin "$BRANCH"

git switch "$BRANCH" 2>/dev/null || \
  git switch -c "$BRANCH" FETCH_HEAD

GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
  git pull --ff-only origin "$BRANCH"

git log -1 --oneline
```

## 2. Create a new isolated temporal-agent Mamba experiment

```bash
BASE=/home/server00/M
cd "$BASE/Codes/Thesis/Lab 2/Codes"

"$BASE/Codes/envs/sharp/bin/python" \
  setup_lab2_temporal_agent_mamba_ablation.py

ROOT=$(cat \
  "$BASE/Codes/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80.txt")
RESULTS=$(cat \
  "$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80.txt")

echo "Code:    $ROOT"
echo "Results: $RESULTS"
cat "$ROOT/EXPERIMENT.txt"
```

Every setup execution creates a new timestamped code and results root. It does
not overwrite a previous run.

## 3. Run in the foreground on all four GPUs

```bash
BASE=/home/server00/M
ROOT=$(cat \
  "$BASE/Codes/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80.txt")

cd "$ROOT/Code"
bash "$ROOT/run_temporal_agent_mamba_4gpu.sh"
```

## 4. Check progress from another terminal

```bash
BASE=/home/server00/M
POINTER="$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80_RUN.txt"

ps -u "$USER" -o pid,ppid,etime,%cpu,%mem,cmd |
  grep "[t]rain.py" || echo "No active SHARP training"

nvidia-smi

if [ -f "$POINTER" ]; then
  RUN=$(cat "$POINTER")
  echo "Run: $RUN"
  tail -20 "$RUN/full_run.log"
fi
```

## 5. Save a one-time terminal log snapshot

This overwrites the destination only when the block is pasted. It does not
keep writing continuously.

```bash
BASE=/home/server00/M
POINTER="$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80_RUN.txt"
DEST="$BASE/Terminal/Terminal_temporal_agent_mamba.txt"

mkdir -p "$(dirname "$DEST")"
RUN=$(cat "$POINTER")
SOURCE="$RUN/full_run.log"

if [ -f "$SOURCE" ]; then
  tr '\r' '\n' < "$SOURCE" > "$DEST"
  sync
  echo "Saved: $DEST"
  ls -lh "$DEST"
else
  echo "ERROR: training log not found: $SOURCE"
fi
```

## 6. Confirm completion and saved checkpoints

```bash
BASE=/home/server00/M
RUN=$(cat \
  "$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80_RUN.txt")

echo "Run: $RUN"
cat "$RUN/COMPLETED.txt"
cat "$RUN/BEST_CHECKPOINT.txt"
cat "$RUN/checkpoint_verification.txt"
find "$RUN/run/checkpoints" -maxdepth 1 -name "*.ckpt" \
  -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort
```
