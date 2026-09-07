# Lab 2 stable temporal-agent Mamba run

The command block below fetches the latest `main` revision using the existing
`madvidd` GitHub CLI token, preserves all earlier code and results, and runs
the corrected experiment in the current foreground terminal.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

run_chunked_temporal_agent_mamba() {
  local BASE REPO GH_BIN ASKPASS STATUS LAUNCHER

  BASE=/home/server00/M
  REPO="$BASE/Codes/Thesis"
  GH_BIN=$(command -v gh)

  if [ -z "$GH_BIN" ]; then
    echo "ERROR: gh is not available."
    return 1
  fi

  "$GH_BIN" auth switch --hostname github.com --user madvidd || return 1
  if [ "$("$GH_BIN" api user --jq .login 2>/dev/null)" != "madvidd" ]; then
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

  LAUNCHER="$REPO/Studies/SHARP/State_Space_Integration/Codes/launch_lab2_temporal_agent_mamba_chunked_58015903.sh"
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

run_chunked_temporal_agent_mamba
STATUS=$?
unset -f run_chunked_temporal_agent_mamba

echo
echo "Terminal remains open."
echo "Final command status: $STATUS"
```

## One-time terminal snapshot

Paste this whenever a new one-time `Terminal.txt` snapshot is needed. It
overwrites the previous snapshot and does not keep writing afterward.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server00/M
POINTER="$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_CHUNKED80_RUN.txt"
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
      echo "===== REAL AV2 PREFLIGHT ====="
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