# Launch or Resume on Lab 3

Paste the entire block in a new terminal. It downloads into a new tool clone,
preserves every earlier run, and reuses this experiment's dedicated pointer on
subsequent invocations. Authentication uses only the lab's Token.txt.
The shell remains open if any step fails. Keep the new training terminal open
while training is active. No commands below stop another process.

```bash
start_seam_combined() (
  set -eu
  set +x
  BASE=/home/server01/M
  PYTHON="$BASE/Codes/envs/seam_av2_mamba_torch211/bin/python"
  TOKEN_FILE="$BASE/Token/Token.txt"
  [ -x "$PYTHON" ] && [ -s "$TOKEN_FILE" ] || {
    echo "ERROR: previous SEAM environment or Token.txt missing."; exit 1;
  }
  export SEAM_GITHUB_TOKEN=$("$PYTHON" - "$TOKEN_FILE" <<'PY'
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
tokens = set()
for encoding in ("utf-8-sig", "utf-16"):
    try:
        text = data.decode(encoding)
    except UnicodeError:
        continue
    tokens.update(re.findall(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text))
if len(tokens) != 1:
    raise SystemExit("ERROR: Token.txt must contain exactly one PAT.")
print(tokens.pop())
PY
  )
  [ -n "$SEAM_GITHUB_TOKEN" ] || exit 1
  for NAME in ${!GIT_@}; do unset "$NAME"; done
  ASKPASS=$(mktemp)
  trap 'rm -f "$ASKPASS"' EXIT
  printf '%s\n' '#!/usr/bin/env sh' 'case "$1" in' \
    '*Username*) printf "%s\n" "madviddd";;' \
    '*Password*) printf "%s\n" "$SEAM_GITHUB_TOKEN";;' \
    'esac' > "$ASKPASS"
  chmod 700 "$ASKPASS"
  export GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 LC_ALL=C
  CLONE="$BASE/Codes/Thesis_SEAM80_Combined_$(date +%Y%m%d-%H%M%S)_$$"
  mkdir -p "$BASE/Codes"
  /usr/bin/git -c credential.helper= clone --depth 1 --branch main \
    --single-branch https://github.com/madvidd/Thesis.git "$CLONE"
  unset SEAM_GITHUB_TOKEN GIT_ASKPASS GIT_TERMINAL_PROMPT
  PACKAGE="$CLONE/Lab 3/SEAM_80_Epoch_Combined_Run"
  BASE="$BASE" bash "$PACKAGE/launch_combined.sh"
)
start_seam_combined
STATUS=$?
unset -f start_seam_combined
echo "Combined-run status: $STATUS"
echo "Terminal remains open."
```

Reuse this block after an interruption. The dedicated pointer identifies the
same immutable experiment; full training resumes at its latest readable epoch
checkpoint. When training is already complete, only unfinished evaluation or
publication is performed. The old baseline and other experiment folders are
not changed, and no old model weights are used to initialise this run.
