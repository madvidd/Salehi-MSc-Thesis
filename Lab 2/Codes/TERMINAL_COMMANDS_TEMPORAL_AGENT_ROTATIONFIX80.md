# Lab 2 rotation-safe temporal-agent Mamba run

This isolated run preserves every earlier code and result directory. It keeps the
same temporal-agent Mamba architecture and 80-epoch four-GPU training settings
as the failed stable-CUDA run. The only numerical runtime change replaces the
inverse of an explicitly orthogonal 2x2 rotation matrix with its exact transpose,
so trajectory conversion no longer creates a cuSOLVER handle at epoch end.

This is not an attention-ablation experiment: all original SHARP attention`ntypes and blocks remain unchanged. The attention-mask compatibility patch only`nresolves PyTorch dtype requirements.

The package also updates deprecated timm imports, gives attention masks matching
dtypes without changing masking semantics, and narrowly filters three known
third-party compatibility warnings. CUDA, NCCL, data, and numerical warnings are
not hidden.

## Fetch and run in the current terminal

This reads the PAT from `/home/server00/M/Token/Token.txt`, verifies the
`madvidd` account and repository access, fast-forwards `main`, and starts the
experiment in the foreground. It never exits the parent terminal.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

run_rotationfix_temporal_agent_mamba() {
  local BASE REPO TOKEN_FILE TOKEN LOGIN ACCESS ASKPASS STATUS STAMP RECOVERY
  local LAUNCHER DIRTY

  BASE=/home/server00/M
  REPO="$BASE/Codes/Thesis"
  TOKEN_FILE="$BASE/Token/Token.txt"
  LAUNCHER="$REPO/Lab 2/Codes/launch_lab2_temporal_agent_mamba_rotationfix_58015903.sh"

  [ -f "$TOKEN_FILE" ] || {
    echo "ERROR: token file is missing: $TOKEN_FILE"
    return 1
  }
  [ -d "$REPO/.git" ] || {
    echo "ERROR: repository clone is missing: $REPO"
    return 1
  }

  TOKEN=$(python3 - "$TOKEN_FILE" <<'PY'
import pathlib
import re
import sys

data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
PY
  )

  [ -n "$TOKEN" ] || {
    echo "ERROR: no GitHub PAT was found in $TOKEN_FILE"
    return 1
  }

  LOGIN=$(GH_TOKEN="$TOKEN" gh api user --jq .login 2>/dev/null)
  ACCESS=$(GH_TOKEN="$TOKEN" gh api repos/madviddd/Thesis --jq .full_name 2>/dev/null)

  if [ "$LOGIN" != "madvidd" ] || [ "$ACCESS" != "madviddd/Thesis" ]; then
    echo "ERROR: PAT verification failed: account=$LOGIN repository=$ACCESS"
    unset TOKEN
    return 1
  fi

  echo "Token verified for $LOGIN with access to $ACCESS."

  export LAB2_GITHUB_TOKEN="$TOKEN"
  unset TOKEN
  ASKPASS=$(mktemp) || return 1
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    '  *Username*) printf "%s\n" "madvidd" ;;' \
    '  *Password*) printf "%s\n" "$LAB2_GITHUB_TOKEN" ;;' \
    'esac' > "$ASKPASS"
  chmod 700 "$ASKPASS"

  cd "$REPO" || {
    rm -f "$ASKPASS"
    unset LAB2_GITHUB_TOKEN
    return 1
  }

  DIRTY=$(git status --porcelain)
  if [ -n "$DIRTY" ]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    RECOVERY="$BASE/Results/Lab2_local_git_before_rotationfix_$STAMP"
    mkdir -p "$RECOVERY"
    git status --short > "$RECOVERY/git_status.txt"
    git diff --binary > "$RECOVERY/tracked_changes.patch"
    git stash push -u -m "Preserve Lab 2 local changes before rotation fix $STAMP"
    STATUS=$?
    if [ "$STATUS" -ne 0 ]; then
      echo "ERROR: local changes could not be preserved. Git update stopped."
      rm -f "$ASKPASS"
      unset LAB2_GITHUB_TOKEN
      return "$STATUS"
    fi
    echo "Local Git changes were preserved in a stash and at: $RECOVERY"
  fi

  git remote set-url origin https://github.com/madviddd/Thesis.git
  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= fetch origin main
  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    if git show-ref --verify --quiet refs/heads/main; then
      git switch main
      STATUS=$?
    else
      git switch -c main FETCH_HEAD
      STATUS=$?
    fi
  fi

  if [ "$STATUS" -eq 0 ]; then
    git merge --ff-only FETCH_HEAD
    STATUS=$?
  fi

  rm -f "$ASKPASS"
  unset LAB2_GITHUB_TOKEN

  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: token-authenticated GitHub update failed."
    return "$STATUS"
  fi

  [ -f "$LAUNCHER" ] || {
    echo "ERROR: corrected launcher is missing: $LAUNCHER"
    return 1
  }

  echo "Installed revision:"
  git log -1 --oneline
  echo

  bash "$LAUNCHER"
  STATUS=$?

  echo
  echo "Foreground experiment returned status: $STATUS"
  return "$STATUS"
}

run_rotationfix_temporal_agent_mamba
STATUS=$?
unset -f run_rotationfix_temporal_agent_mamba

echo
echo "Terminal remains open."
echo "Final command status: $STATUS"
```

## Expected startup checks

Before full training starts, the output must contain all of these markers:

```text
ROTATION_TRANSPOSE_PATCH_ACTIVE=True
ORIGINAL_SHARP_ATTENTION_MASK_COMPATIBILITY_ACTIVE=True
DEPRECATED_TIMM_IMPORTS_PRESENT=False
SHARP_MODEL_IMPORT_OK=True
TEMPORAL_AGENT_MAMBA_SMOKE_TEST_OK
MAMBA_ACTIVE=True
CAUSAL_CONV_BACKEND=torch_cuda_cudnn_conv1d
SELECTIVE_SCAN_BACKEND=fused_selective_scan_cuda
```

The four-GPU preflight must complete 256 real AV2 training batches. The full
run starts only after that preflight succeeds.

## One-time Terminal.txt snapshot

Paste this whenever a new snapshot is needed. It overwrites `Terminal.txt` once
and does not keep writing.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server00/M
POINTER="$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_ROTATIONFIX80_RUN.txt"
DEST="$BASE/Terminal/Terminal.txt"

mkdir -p "$(dirname "$DEST")"

if [ ! -s "$POINTER" ]; then
  echo "ERROR: run pointer not found: $POINTER"
else
  RUN=$(tr -d '\r\n' < "$POINTER")
  {
    echo "Lab 2 rotation-safe temporal-agent Mamba snapshot"
    echo "Saved: $(date)"
    echo "Run: $RUN"
    echo

    if [ -f "$RUN/RUN_SETTINGS.txt" ]; then
      echo "===== RUN SETTINGS ====="
      cat "$RUN/RUN_SETTINGS.txt"
      echo
    fi

    if [ -f "$RUN/preflight/preflight.log" ]; then
      echo "===== FOUR-GPU PREFLIGHT ====="
      tr '\r' '\n' < "$RUN/preflight/preflight.log"
      echo
    fi

    if [ -f "$RUN/full_run.log" ]; then
      echo "===== FULL FOUR-GPU TRAINING ====="
      tr '\r' '\n' < "$RUN/full_run.log"
      echo
    fi

    echo "===== SAVED CHECKPOINTS ====="
    find "$RUN" -name "*.ckpt" \
      -printf "%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n" | sort
  } > "$DEST"

  sync
  echo "Saved one-time Lab 2 snapshot:"
  echo "$DEST"
  ls -lh "$DEST"
fi

echo "Terminal remains open."
```
