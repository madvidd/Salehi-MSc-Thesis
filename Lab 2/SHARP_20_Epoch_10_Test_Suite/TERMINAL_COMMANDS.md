# Lab 2 Terminal Commands

All blocks leave the current terminal open. Run the suite in the foreground.

## 1. Preserve the current Lab 2 run before closing its terminal

Run this in a separate Lab 2 terminal while training is still active, or in the current terminal after training has returned to the prompt:

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server00/M
DEST="$BASE/Terminal/Terminal.txt"
STAMP=$(date +%Y%m%d-%H%M%S)
ARCHIVE="$BASE/Terminal/Before_10_Test_Suite_$STAMP"
mkdir -p "$ARCHIVE" "$(dirname "$DEST")"

RUN=""
for POINTER in \
  "$BASE/Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_ROTATIONFIX80_RUN.txt" \
  "$BASE/Results/LATEST_SHARP_AV2_MAMBA_FUSED80_RUN.txt"
do
  if [ -s "$POINTER" ]; then
    CANDIDATE=$(tr -d '\r\n' < "$POINTER")
    if [ -d "$CANDIDATE" ]; then
      RUN="$CANDIDATE"
      break
    fi
  fi
done

if [ -z "$RUN" ]; then
  RUN=$(find "$BASE/Results" -maxdepth 4 -type f \
    \( -name full_run.log -o -name train.log \) \
    -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
fi

SOURCE=""
[ -s "$RUN/full_run.log" ] && SOURCE="$RUN/full_run.log"
[ -z "$SOURCE" ] && [ -s "$RUN/train.log" ] && SOURCE="$RUN/train.log"

if [ -n "$SOURCE" ]; then
  tr '\r' '\n' < "$SOURCE" > "$DEST"
  cp -p "$DEST" "$ARCHIVE/Terminal.txt"
  find "$RUN" -name '*.ckpt' \
    -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' | sort \
    > "$ARCHIVE/CHECKPOINTS.txt"
  printf 'saved=%s\nrun=%s\nsource=%s\n' \
    "$(date --iso-8601=seconds)" "$RUN" "$SOURCE" \
    > "$ARCHIVE/SNAPSHOT_INFO.txt"
  sync
  echo "CURRENT_RUN_SAVED"
  echo "Terminal snapshot: $DEST"
  echo "Archive metadata:  $ARCHIVE"
  ls -lh "$DEST" "$ARCHIVE/CHECKPOINTS.txt"
else
  echo "ERROR: no current Lab 2 training log was found."
fi

pgrep -af '[t]rain.py' || echo 'No SHARP training process is active.'
echo 'Terminal remains open.'
```

## 2. Update the Thesis repository using Token.txt

```bash
set +e
set +u
set +o pipefail 2>/dev/null

BASE=/home/server00/M
REPO="$BASE/Codes/Thesis"
TOKEN_FILE="$BASE/Token/Token.txt"
TOKEN=$(python3 - "$TOKEN_FILE" <<'PY'
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
PY
)

LOGIN=$(GH_TOKEN="$TOKEN" gh api user --jq .login 2>/dev/null)
echo "GitHub token account: $LOGIN"

if [ "$LOGIN" = "madviddd" ]; then
  export LAB2_GITHUB_TOKEN="$TOKEN"
  ASKPASS=$(mktemp)
  printf '%s\n' '#!/usr/bin/env bash' \
    'case "$1" in' \
    ' *Username*) printf "%s\n" "madviddd" ;;' \
    ' *Password*) printf "%s\n" "$LAB2_GITHUB_TOKEN" ;;' \
    'esac' > "$ASKPASS"
  chmod 700 "$ASKPASS"
  cd "$REPO"
  /usr/bin/git remote set-url origin https://github.com/madvidd/Thesis.git
  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    /usr/bin/git -c credential.helper= fetch origin main
  STATUS=$?
  if [ "$STATUS" -eq 0 ]; then
    /usr/bin/git switch main
    /usr/bin/git merge --autostash FETCH_HEAD
    STATUS=$?
  fi
  rm -f "$ASKPASS"
  unset LAB2_GITHUB_TOKEN
else
  echo 'ERROR: Token.txt is not a valid madviddd token.'
  STATUS=1
fi
unset TOKEN
echo "Update status: $STATUS"
echo 'Terminal remains open.'
```

## 3. Prepare a new isolated experiment

```bash
cd "/home/server00/M/Codes/Thesis/Lab 2/SHARP_20_Epoch_10_Test_Suite"

/home/server00/M/Codes/envs/sharp/bin/python \
  setup_lab2_sharp_20epoch_10test.py

ROOT=$(tr -d '\r\n' < \
  /home/server00/M/Codes/LATEST_SHARP_AV2_20EPOCH_10TEST.txt)

echo "Experiment root: $ROOT"
ls -lh "$ROOT/run_10_test_suite.sh"
echo 'Terminal remains open.'
```

## 4. Run or resume all ten tests in sequence

```bash
ROOT=$(tr -d '\r\n' < \
  /home/server00/M/Codes/LATEST_SHARP_AV2_20EPOCH_10TEST.txt)

cd "$ROOT/Code"
bash "$ROOT/run_10_test_suite.sh"

STATUS=$?
echo "Suite command status: $STATUS"
echo 'Terminal remains open.'
```

If a variant fails or the terminal disconnects, run block 4 again. Completed variants are skipped, and the interrupted variant resumes from its most recent completed epoch.

## 5. Check status from another terminal without interrupting training

```bash
BASE=/home/server00/M
ROOT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_AV2_20EPOCH_10TEST.txt")
RESULTS=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_AV2_20EPOCH_10TEST.txt")

"$BASE/Codes/envs/sharp/bin/python" \
  "$ROOT/suite_status.py" "$RESULTS"

pgrep -af '[t]rain.py' || echo 'No SHARP training process is active.'
nvidia-smi
echo 'Terminal remains open.'
```
