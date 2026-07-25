# Lab 2 terminal commands

## 1. Fetch this branch without username/password prompts

```bash
BASE=/home/server00/M
REPO="$BASE/Codes/Thesis"
BRANCH=lab2-temporal-agent-mamba-ablation

cd "$REPO"
gh auth switch --hostname github.com --user madviddd

ASKPASS="$REPO/.git/gh-token-askpass.sh"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'case "$1" in' \
  '  *Username*) printf "%s\n" "madviddd" ;;' \
  '  *Password*) gh auth token --hostname github.com ;;' \
  'esac' > "$ASKPASS"
chmod 700 "$ASKPASS"

GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
  git fetch origin "$BRANCH"

git switch "$BRANCH" 2>/dev/null || \
  git switch -c "$BRANCH" FETCH_HEAD

GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
  git pull --ff-only origin "$BRANCH"
```

## 2. Create isolated experiment copies

```bash
BASE=/home/server00/M
cd "$BASE/Codes/Thesis/Lab 2/Codes"

"$BASE/Codes/envs/sharp/bin/python" \
  setup_lab2_temporal_agent_mamba_ablation.py

ROOT=$(cat \
  "$BASE/Codes/LATEST_SHARP_AV2_TEMPORAL_MAMBA_ABLATION.txt")
RESULTS=$(cat \
  "$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_MAMBA_ABLATION.txt")

echo "Code:    $ROOT"
echo "Results: $RESULTS"
cat "$ROOT/EXPERIMENT.txt"
```

## 3A. Recommended: run the exact control first

```bash
BASE=/home/server00/M
ROOT=$(cat \
  "$BASE/Codes/LATEST_SHARP_AV2_TEMPORAL_MAMBA_ABLATION.txt")

cd "$ROOT/baseline_control/Code"
bash "$ROOT/run_baseline_control_4gpu.sh"
```

After the control finishes, run temporal-agent Mamba:

```bash
BASE=/home/server00/M
ROOT=$(cat \
  "$BASE/Codes/LATEST_SHARP_AV2_TEMPORAL_MAMBA_ABLATION.txt")

cd "$ROOT/temporal_agent_mamba/Code"
bash "$ROOT/run_temporal_agent_mamba_4gpu.sh"
```

## 3B. Alternative: run both sequentially

Do not use this if either individual runner is already active.

```bash
BASE=/home/server00/M
ROOT=$(cat \
  "$BASE/Codes/LATEST_SHARP_AV2_TEMPORAL_MAMBA_ABLATION.txt")

bash "$ROOT/run_control_then_temporal.sh"
```

## 4. Check status from another terminal

```bash
ps -u "$USER" -o pid,ppid,etime,%cpu,%mem,cmd |
  grep "[t]rain.py" || echo "No active SHARP training"

nvidia-smi

for POINTER in \
  /home/server00/M/Results/LATEST_SHARP_AV2_BASELINE_CONTROL80_RUN.txt \
  /home/server00/M/Results/LATEST_SHARP_AV2_TEMPORAL_MAMBA80_RUN.txt
do
  if [ -f "$POINTER" ]; then
    RUN=$(cat "$POINTER")
    echo
    echo "Run: $RUN"
    tail -20 "$RUN/full_run.log"
  fi
done
```

## 5. Compare completed validations

```bash
BASE=/home/server00/M
ROOT=$(cat \
  "$BASE/Codes/LATEST_SHARP_AV2_TEMPORAL_MAMBA_ABLATION.txt")

"$BASE/Codes/envs/sharp/bin/python" \
  "$ROOT/compare_completed_runs.py"

cat "$BASE/Results/SHARP_AV2_TEMPORAL_MAMBA_COMPARISON.csv"
```

## 6. One-time log snapshots

This overwrites each destination only when the block is run. It does not keep
writing continuously.

```bash
BASE=/home/server00/M
mkdir -p "$BASE/Terminal"

for ITEM in \
  "LATEST_SHARP_AV2_BASELINE_CONTROL80_RUN.txt:Terminal_baseline_control.txt" \
  "LATEST_SHARP_AV2_TEMPORAL_MAMBA80_RUN.txt:Terminal_temporal_agent_mamba.txt"
do
  POINTER=${ITEM%%:*}
  DEST=${ITEM#*:}
  POINTER="$BASE/Results/$POINTER"

  if [ -f "$POINTER" ]; then
    RUN=$(cat "$POINTER")
    if [ -f "$RUN/full_run.log" ]; then
      tr '\r' '\n' < "$RUN/full_run.log" > "$BASE/Terminal/$DEST"
      echo "Saved: $BASE/Terminal/$DEST"
      ls -lh "$BASE/Terminal/$DEST"
    fi
  fi
done
```
