# Lab 2 temporal-agent Mamba compatibility launch

Paste this block into one Lab 2 terminal. It disables interactive-shell error
exit, downloads the latest `main` branch, stops only earlier Lab 2 SHARP Mamba
jobs, validates an isolated NVIDIA 580.159.03 user-space environment, creates
new timestamped code and result directories, and starts training in the
foreground. It does not use `source`, `exec`, or `exit` in the interactive
terminal.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

run_new_temporal_agent_mamba() {
  local BASE REPO LAUNCHER STATUS

  BASE=/home/server00/M
  REPO="$BASE/Codes/Thesis"

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

  LAUNCHER="$REPO/Studies/SHARP/State_Space_Integration/Codes/launch_lab2_temporal_agent_mamba_58015903.sh"
  test -f "$LAUNCHER" || {
    echo "ERROR: launcher is missing: $LAUNCHER"
    return 1
  }

  bash "$LAUNCHER"
  STATUS=$?

  echo
  echo "Foreground launcher returned status: $STATUS"
  return "$STATUS"
}

run_new_temporal_agent_mamba
STATUS=$?
unset -f run_new_temporal_agent_mamba

echo
echo "Terminal remains open."
echo "Final command status: $STATUS"
```
