# Lab 2 stable-CUDA temporal-agent Mamba run

This is the third isolated temporal-agent experiment. It preserves all earlier
code and results. It keeps the exact temporal-agent architecture and training
configuration from the chunked experiment, but disables only the unstable
optional `causal_conv1d_cuda` path. The short convolution runs on the GPU
through PyTorch/cuDNN, while the official fused CUDA selective scan remains
active.

The launcher first runs 256 real AV2 training batches using the exact four-GPU
DDP and SyncBatchNorm path with synchronous CUDA error reporting. The full
80-epoch run starts only if that preflight passes beyond the previous batch-122
failure point.

## Fetch and run in the current terminal

```bash
set +e
set +u
set +o pipefail 2>/dev/null

run_stablecuda_temporal_agent_mamba() {
  local BASE REPO GH_BIN ASKPASS STATUS LAUNCHER LOGIN

  BASE=/home/server00/M
  REPO="$BASE/Codes/Thesis"
  GH_BIN=$(command -v gh)

  if [ -z "$GH_BIN" ]; then
    echo "ERROR: gh is not available."
    return 1
  fi

  "$GH_BIN" auth switch --hostname github.com --user madvidd || return 1
  LOGIN=$("$GH_BIN" api user --jq .login 2>/dev/null)
  if [ "$LOGIN" != "madvidd" ]; then
    echo "ERROR: the active GitHub token is not for madvidd."
    return 1
  fi

  cd "$REPO" || {
    echo "ERROR: repository not found: $REPO"
    return 1
  }

  ASKPASS=$(mktemp) || return 1
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    '  *Username*) printf "%s\n" "madvidd" ;;' \
    "  *Password*) \"$GH_BIN\" auth token --hostname github.com ;;" \
    'esac' > "$ASKPASS"
  chmod 700 "$ASKPASS"

  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= fetch origin main
  STATUS=$?
  rm -f "$ASKPASS"

  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: token-authenticated GitHub fetch failed."
    return "$STATUS"
  fi

  if git show-ref --verify --quiet refs/heads/main; then
    git switch main || return 1
    git merge --ff-only FETCH_HEAD || return 1
  else
    git switch -c main FETCH_HEAD || return 1
  fi

  LAUNCHER="$REPO/Studies/SHARP/State_Space_Integration/Codes/launch_lab2_temporal_agent_mamba_stablecuda_58015903.sh"
  if [ ! -f "$LAUNCHER" ]; then
    echo "ERROR: launcher is missing: $LAUNCHER"
    return 1
  fi

  bash "$LAUNCHER"
  STATUS=$?
  echo
  echo "Foreground experiment returned status: $STATUS"
  return "$STATUS"
}

run_stablecuda_temporal_agent_mamba
STATUS=$?
unset -f run_stablecuda_temporal_agent_mamba

echo
echo "Terminal remains open."
echo "Final command status: $STATUS"
```

## One-time Terminal.txt snapshot

Paste this whenever a new snapshot is needed. It overwrites `Terminal.txt`
once and does not continue writing.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server00/M
POINTER="$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_STABLECUDA80_RUN.txt"
DEST="$BASE/Terminal/Terminal.txt"

mkdir -p "$(dirname "$DEST")"

if [ ! -s "$POINTER" ]; then
  echo "ERROR: run pointer not found: $POINTER"
else
  RUN=$(cat "$POINTER")
  {
    echo "Run: $RUN"
    echo
    if [ -f "$RUN/preflight/preflight.log" ]; then
      echo "===== EXACT 256-BATCH FOUR-GPU PREFLIGHT ====="
      tr '\r' '\n' < "$RUN/preflight/preflight.log"
      echo
    fi
    if [ -f "$RUN/full_run.log" ]; then
      echo "===== FULL FOUR-GPU TRAINING ====="
      tr '\r' '\n' < "$RUN/full_run.log"
    fi
  } > "$DEST"
  sync
  echo "Saved one-time Lab 2 snapshot:"
  echo "$DEST"
  ls -lh "$DEST"
fi

echo "Terminal remains open."
```
