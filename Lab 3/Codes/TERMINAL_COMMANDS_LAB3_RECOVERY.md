# Lab 3 CUDA recovery, GitHub cleanup, and non-baseline restart

This recovery uses no `sudo`. It removes the `madvidd` GitHub CLI credential,
authenticates the token in `/home/server01/M/Token/Token.txt` as `madviddd`,
performs a real pull/integrate/push test against `madvidd/Trading_Bot`, preserves
the current partial attention results, installs a user-owned PyTorch 2.8 CUDA
12.6 environment, stress-tests all three custom attention variants on all three
GPUs, and restarts only:

1. `qknorm`
2. `talking_heads`
3. `qknorm_talking_heads`

`baseline_mha` is explicitly excluded.

Run the following block in a new SSH terminal. It deliberately avoids `exit`, so
validation failures return to the SSH prompt.

```bash
set +e
set +u
set +o pipefail 2>/dev/null

apply_lab3_recovery() {
  BASE=/home/server01/M
  TOKEN_FILE="$BASE/Token/Token.txt"
  PYTHON="$BASE/Codes/AV2/miniforge3/bin/python"
  REPO=$(tr -d '\r\n' < "$BASE/Codes/LATEST_THESIS_LAB3_CLONE.txt")
  BRANCH=lab3-sharp-attention-ablation

  [ -f "$TOKEN_FILE" ] || {
    echo "FAIL: missing $TOKEN_FILE"
    return 1
  }
  [ -x "$PYTHON" ] || {
    echo "FAIL: missing $PYTHON"
    return 1
  }
  [ -d "$REPO/.git" ] || {
    echo "FAIL: missing Thesis clone: $REPO"
    return 1
  }
  command -v gh >/dev/null 2>&1 || {
    echo "FAIL: gh is not installed"
    return 1
  }

  chmod 600 "$TOKEN_FILE"
  TOKEN=$("$PYTHON" -c '
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
' "$TOKEN_FILE")
  [ -n "$TOKEN" ] || {
    echo "FAIL: no PAT found in Token.txt"
    return 1
  }

  gh auth logout --hostname github.com --user madvidd >/dev/null 2>&1 || true
  printf 'protocol=https\nhost=github.com\nusername=madvidd\n\n' \
    | git credential reject >/dev/null 2>&1 || true
  gh auth logout --hostname github.com --user madviddd >/dev/null 2>&1 || true
  printf '%s\n' "$TOKEN" | gh auth login --hostname github.com --with-token || {
    echo "FAIL: replacement token was rejected"
    unset TOKEN
    return 1
  }
  unset TOKEN

  gh auth switch --hostname github.com --user madviddd || return 1
  gh auth setup-git || return 1
  ACTIVE=$(gh api user --jq .login 2>/dev/null)
  [ "$ACTIVE" = madviddd ] || {
    echo "FAIL: active account is $ACTIVE"
    return 1
  }
  echo "PASS: active GitHub account is $ACTIVE"

  git -C "$REPO" remote set-url origin https://github.com/madvidd/Thesis.git
  if ! git -C "$REPO" diff --quiet -- \
    "Lab 3/Codes/run_remaining_attention_suite.sh"; then
    BACKUP="$HOME/lab3_previous_launcher_$(date +%Y%m%d-%H%M%S).patch"
    git -C "$REPO" diff -- \
      "Lab 3/Codes/run_remaining_attention_suite.sh" > "$BACKUP"
    git -C "$REPO" restore --source=HEAD -- \
      "Lab 3/Codes/run_remaining_attention_suite.sh"
    echo "Preserved old launcher changes at: $BACKUP"
  fi

  git -C "$REPO" fetch --prune origin || return 1
  git -C "$REPO" switch "$BRANCH" || return 1
  git -C "$REPO" pull --rebase origin "$BRANCH" || return 1
  bash "$REPO/Lab 3/Codes/recover_and_restart_lab3.sh"
  return $?
}

apply_lab3_recovery
STATUS=$?
unset -f apply_lab3_recovery

echo
echo "Final recovery status: $STATUS"
echo "The SSH shell remains open."
```
