# Lab 2 terminal commands

These commands never call `exit` and intentionally disable interactive-shell
`errexit`. Run the generated launcher with `bash`; do not use `source`. The
terminal remains open and returns to its prompt after training succeeds or
fails.

## Stop only a previous temporal-agent Mamba run

This targets the generated temporal-agent Mamba runner and training processes.
It does not use `pkill` against every Python process and never closes the
interactive terminal.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

PATTERN='SHARP_AV2_TEMPORAL_AGENT_MAMBA8[0]|run_temporal_agent_mamba_4gpu[.]sh'
PIDS=$(pgrep -u "$USER" -f "$PATTERN")

if [ -n "$PIDS" ]; then
  echo "Stopping temporal-agent Mamba processes: $PIDS"
  kill -INT $PIDS 2>/dev/null
  sleep 10

  PIDS=$(pgrep -u "$USER" -f "$PATTERN")
  if [ -n "$PIDS" ]; then
    echo "Processes still active; sending TERM: $PIDS"
    kill -TERM $PIDS 2>/dev/null
    sleep 10
  fi
else
  echo "No previous temporal-agent Mamba process is active."
fi

PIDS=$(pgrep -u "$USER" -f "$PATTERN")
if [ -n "$PIDS" ]; then
  echo "Processes still active; sending targeted KILL: $PIDS"
  kill -KILL $PIDS 2>/dev/null
  sleep 3
fi

PIDS=$(pgrep -u "$USER" -f "$PATTERN")
if [ -n "$PIDS" ]; then
  echo "WARNING: these targeted processes remain: $PIDS"
  ps -o pid,ppid,pgid,etime,%cpu,%mem,cmd -p $PIDS
else
  echo "Previous temporal-agent Mamba run is stopped."
fi

echo "Terminal remains open."
```

## Pull, create and run in the same foreground terminal

```bash
set +e
set +u
set +o pipefail 2>/dev/null

run_lab2_temporal_agent_mamba() {
  local BASE REPO SETUP ROOT RESULTS STATUS

  BASE=/home/server00/M
  REPO="$BASE/Codes/Thesis"
  SETUP="$REPO/Lab 2/Codes/setup_lab2_temporal_agent_mamba_ablation.py"

  cd "$REPO" || {
    echo "ERROR: repository not found: $REPO"
    return 1
  }

  git fetch origin main || {
    echo "ERROR: could not fetch GitHub main"
    return 1
  }

  if git show-ref --verify --quiet refs/heads/main; then
    git switch main || return 1
    git merge --ff-only FETCH_HEAD || return 1
  else
    git switch -c main FETCH_HEAD || return 1
  fi

  test -f "$SETUP" || {
    echo "ERROR: setup file is missing: $SETUP"
    return 1
  }

  test -d "$BASE/Codes/SHARP/Code" || {
    echo "ERROR: original SHARP source is missing"
    return 1
  }

  test -x "$BASE/Codes/envs/sharp/bin/python" || {
    echo "ERROR: SHARP Python environment is missing"
    return 1
  }

  test -d "$BASE/Datasets/AV2/sharp_processed/train" || {
    echo "ERROR: processed AV2 train data is missing"
    return 1
  }

  test -d "$BASE/Datasets/AV2/sharp_processed/val" || {
    echo "ERROR: processed AV2 validation data is missing"
    return 1
  }

  cd "$REPO/Lab 2/Codes" || return 1

  "$BASE/Codes/envs/sharp/bin/python" \
    setup_lab2_temporal_agent_mamba_ablation.py || {
      echo "ERROR: experiment setup failed"
      return 1
    }

  ROOT=$(cat \
    "$BASE/Codes/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80.txt") || return 1
  RESULTS=$(cat \
    "$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80.txt") || return 1

  echo "New code:    $ROOT"
  echo "New results: $RESULTS"
  cat "$ROOT/EXPERIMENT.txt"

  cd "$ROOT/Code" || return 1

  echo "Starting foreground training on all four GPUs."
  echo "Do not close this terminal while training is active."
  bash "$ROOT/run_temporal_agent_mamba_4gpu.sh"
  STATUS=$?

  echo
  echo "Training command finished with status: $STATUS"
  echo "The terminal remains open."
  return "$STATUS"
}

run_lab2_temporal_agent_mamba
unset -f run_lab2_temporal_agent_mamba
```

The `git pull` only downloads the code already published from the laptop. It
does not upload or push anything from Lab 2. Every setup execution creates new
timestamped code and results directories and preserves all previous runs.

## Check progress from another terminal

```bash
set +e
BASE=/home/server00/M
POINTER="$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80_RUN.txt"

ps -u "$USER" -o pid,ppid,etime,%cpu,%mem,cmd |
  grep "[t]rain.py" || echo "No active SHARP training"

nvidia-smi

if [ -f "$POINTER" ]; then
  RUN=$(cat "$POINTER")
  echo "Run: $RUN"
  tail -20 "$RUN/full_run.log"
else
  echo "No run pointer exists yet."
fi
```

## Save a one-time terminal log snapshot

This overwrites the destination only when the block is pasted. It does not
keep writing continuously.

```bash
set +e
BASE=/home/server00/M
POINTER="$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80_RUN.txt"
DEST="$BASE/Terminal/Terminal_temporal_agent_mamba.txt"

mkdir -p "$(dirname "$DEST")"

if [ -f "$POINTER" ]; then
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
else
  echo "ERROR: run pointer not found: $POINTER"
fi
```

## Confirm completion and saved checkpoints

```bash
set +e
BASE=/home/server00/M
POINTER="$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA80_RUN.txt"

if [ -f "$POINTER" ]; then
  RUN=$(cat "$POINTER")
  echo "Run: $RUN"
  cat "$RUN/COMPLETED.txt" 2>/dev/null ||
    echo "The run has not written its completion marker yet."
  cat "$RUN/BEST_CHECKPOINT.txt" 2>/dev/null || true
  cat "$RUN/checkpoint_verification.txt" 2>/dev/null || true
  find "$RUN/run/checkpoints" -maxdepth 1 -name "*.ckpt" \
    -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort
else
  echo "ERROR: run pointer not found: $POINTER"
fi
```