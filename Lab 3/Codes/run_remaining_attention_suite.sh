#!/usr/bin/env bash

# Continue the existing Lab 3 SHARP attention ablation without rerunning baseline.
# Each variant is run in the foreground, archived locally, published without large
# files, merged into main, and followed by the next variant.

set -uo pipefail

BASE=/home/server01/M
TOKEN_FILE="$BASE/Token/Token.txt"
ROOT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")
RESULTS_ROOT=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
ENV="$BASE/Codes/AV2/envs/sharp_av2"
CONDA="$BASE/Codes/AV2/miniforge3/bin/conda"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
WORK_BRANCH=lab3-sharp-attention-ablation
VARIANTS=(qknorm talking_heads qknorm_talking_heads)
SUITE_STAMP=$(date +%Y%m%d-%H%M%S)
SUITE_LOG="$RESULTS_ROOT/remaining_attention_suite_$SUITE_STAMP.log"

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$SUITE_LOG") 2>&1

echo "REMAINING_ATTENTION_SUITE_START=$(date --iso-8601=seconds)"
echo "ROOT=$ROOT"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "SUITE_LOG=$SUITE_LOG"
echo "VARIANTS=${VARIANTS[*]}"
echo "BASELINE_MHA_IS_EXPLICITLY_EXCLUDED=True"

for required in "$TOKEN_FILE" "$ROOT/run_variant.sh" "$ENV/bin/python"; do
  if [[ ! -e "$required" ]]; then
    echo "FATAL: required path is missing: $required"
    echo "Terminal remains open after this script returns."
    exit 1
  fi
done

chmod 600 "$TOKEN_FILE"
TOKEN=$("$ENV/bin/python" -c '
import pathlib, re, sys
data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
' "$TOKEN_FILE")

if [[ -z "$TOKEN" ]]; then
  echo "FATAL: no GitHub PAT was found in $TOKEN_FILE"
  exit 1
fi

USER_JSON=$(mktemp)
USER_HTTP=$(curl -sS -o "$USER_JSON" -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/user)
LOGIN=$("$ENV/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1])).get("login",""))' \
  "$USER_JSON" 2>/dev/null)
rm -f "$USER_JSON"

if [[ "$USER_HTTP" != 200 || "$LOGIN" != madvidd ]]; then
  echo "FATAL: GitHub authentication failed: HTTP=$USER_HTTP account=$LOGIN"
  unset TOKEN
  exit 1
fi
echo "GITHUB_AUTHENTICATED_ACCOUNT=$LOGIN"

archive_variant() {
  local variant=$1
  local run_status=$2
  local stamp archive out code file rel

  stamp=$(date +%Y%m%d-%H%M%S)
  archive="$BASE/Results/Archives/Lab3_SHARP_Attention_${variant}_$stamp"
  out="$RESULTS_ROOT/$variant"
  code="$ROOT/variants/$variant/Code"
  mkdir -p "$archive/configuration" "$archive/reproducibility"

  {
    echo "variant=$variant"
    echo "saved=$(date --iso-8601=seconds)"
    echo "training_exit_code=$run_status"
    echo "code=$code"
    echo "results=$out"
    echo "epochs=80"
    echo "gpus=3"
    echo "batch_per_gpu=8"
    echo "global_batch=24"
    echo "seed=2333"
    echo "sync_batchnorm=true"
    echo "baseline_rerun=false"
  } > "$archive/MANIFEST.txt"

  [[ -f "$out/metrics.json" ]] && cp -p "$out/metrics.json" "$archive/FINAL_METRICS.json"
  [[ -f "$out/best_checkpoint.txt" ]] && cp -p "$out/best_checkpoint.txt" "$archive/"
  [[ -f "$out/COMPLETE" ]] && cp -p "$out/COMPLETE" "$archive/"
  [[ -f "$ROOT/experiment_manifest.json" ]] && cp -p "$ROOT/experiment_manifest.json" "$archive/"

  find "$out" -name '*.ckpt' -type f \
    -printf '%TY-%Tm-%Td %TH:%TM  %s bytes  %p\n' 2>/dev/null \
    | sort > "$archive/CHECKPOINTS.txt"

  {
    echo "===== TRAINING STATUS ====="
    echo "exit_code=$run_status"
    echo
    echo "===== METRICS ====="
    [[ -f "$out/metrics.json" ]] && cat "$out/metrics.json"
    echo
    echo "===== FINAL TRAIN LOG LINES ====="
    [[ -f "$out/train.log" ]] && tr '\r' '\n' < "$out/train.log" | tail -250
    echo
    echo "===== FINAL EVALUATION LOG LINES ====="
    [[ -f "$out/eval.log" ]] && tr '\r' '\n' < "$out/eval.log" | tail -250
  } > "$archive/Terminal_Summary.txt"

  while IFS= read -r -d '' file; do
    rel=${file#"$out/"}
    mkdir -p "$archive/configuration/$(dirname "$rel")"
    cp -p "$file" "$archive/configuration/$rel"
  done < <(find "$out" -path '*/.hydra/*.yaml' -type f -print0 2>/dev/null)

  for file in \
    "$code/src/model/layers/attention_variants.py" \
    "$code/src/model/layers/custom_transformer_blocks.py" \
    "$code/src/model/layers/transformer_blocks.py" \
    "$ROOT/run_variant.sh" \
    "$ROOT/experiment_manifest.json"
  do
    [[ -f "$file" ]] && cp -p "$file" "$archive/reproducibility/$(basename "$file")"
  done

  "$ENV/bin/python" -m pip freeze > "$archive/pip-freeze.txt" 2>&1
  [[ -x "$CONDA" ]] && "$CONDA" list -p "$ENV" > "$archive/conda-list.txt" 2>&1
  nvidia-smi > "$archive/nvidia-smi.txt" 2>&1

  if ! tar -czf "$archive/results_and_checkpoints.tar.gz" \
    -C "$RESULTS_ROOT" "$variant"; then
    echo "Failed to archive results for $variant" >&2
    return 1
  fi
  if ! tar -czf "$archive/code.tar.gz" \
    -C "$ROOT/variants/$variant" Code; then
    echo "Failed to archive code for $variant" >&2
    return 1
  fi
  if ! sha256sum "$archive"/*.tar.gz > "$archive/SHA256SUMS.txt"; then
    echo "Failed to checksum archives for $variant" >&2
    return 1
  fi

  printf '%s\n' "$archive" \
    > "$BASE/Results/LATEST_LAB3_${variant^^}_ARCHIVE.txt"
  echo "$archive"
}

publish_and_merge() {
  local variant=$1
  local archive=$2
  local stamp rel dest askpass status file item

  stamp=$(basename "$archive")
  rel="Lab 3/Main_Results/Attention_Experiments/$stamp"
  dest="$REPO/$rel"
  askpass=$(mktemp)
  status=0

  export LAB3_GITHUB_TOKEN="$TOKEN"
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    '  *Username*) printf "%s\n" "madvidd" ;;' \
    '  *Password*) printf "%s\n" "$LAB3_GITHUB_TOKEN" ;;' \
    'esac' > "$askpass"
  chmod 700 "$askpass"

  cd "$REPO" || return 1
  git remote set-url origin https://github.com/madviddd/Thesis.git
  GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= fetch origin \
    "$WORK_BRANCH:refs/remotes/origin/$WORK_BRANCH" \
    "main:refs/remotes/origin/main" || status=$?

  if (( status == 0 )); then
    git switch "$WORK_BRANCH" || status=$?
  fi
  if (( status == 0 )); then
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
      git -c credential.helper= pull --ff-only origin "$WORK_BRANCH" || status=$?
  fi

  if (( status == 0 )); then
    mkdir -p "$dest"
    while IFS= read -r -d '' file; do
      item=${file#"$archive/"}
      case "$item" in
        *.ckpt|*.tar.gz|*.log|Terminal.txt|*/Terminal.txt|*Token.txt) continue ;;
      esac
      mkdir -p "$dest/$(dirname "$item")"
      cp -p "$file" "$dest/$item"
    done < <(find "$archive" -type f -size -10M -print0)

    printf '%s\n' '*.ckpt' '*.tar.gz' '*.log' 'Terminal.txt' '*Token.txt' \
      > "$dest/.gitignore"
    {
      echo "# Lab 3 SHARP attention result: $variant"
      echo
      echo "The complete archive, checkpoints and full logs remain on Lab 3:"
      echo
      echo "\`$archive\`"
    } > "$dest/README.md"

    if grep -RIlE 'github_pat_|ghp_' "$dest" >/dev/null 2>&1; then
      echo "PUBLISH_ERROR[$variant]: credential text detected"
      status=1
    elif find "$dest" -type f -size +10M | grep -q .; then
      echo "PUBLISH_ERROR[$variant]: file larger than 10 MB detected"
      status=1
    fi
  fi

  if (( status == 0 )); then
    git config user.name "Seyed Mohammad Salehi"
    git config user.email "madvidd@users.noreply.github.com"
    git add -- "$rel"
    if ! git diff --cached --quiet -- "$rel"; then
      git commit -m "Add Lab 3 $variant attention results" -- "$rel" || status=$?
    fi
  fi
  if (( status == 0 )); then
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
      git -c credential.helper= push origin "$WORK_BRANCH" || status=$?
  fi

  if (( status == 0 )); then
    if git show-ref --verify --quiet refs/heads/main; then
      git switch main || status=$?
    else
      git switch --track -c main origin/main || status=$?
    fi
  fi
  if (( status == 0 )); then
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
      git -c credential.helper= pull --ff-only origin main || status=$?
  fi
  if (( status == 0 )); then
    git merge --no-edit "$WORK_BRANCH" || status=$?
  fi
  if (( status == 0 )); then
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
      git -c credential.helper= push origin main || status=$?
  fi

  if (( status != 0 )); then
    git merge --abort 2>/dev/null || true
    echo "PUBLISH_OR_MERGE_FAILED[$variant]=$status"
  else
    echo "PUBLISH_AND_MERGE_COMPLETE[$variant]"
  fi

  git switch "$WORK_BRANCH" >/dev/null 2>&1 || true
  if (( status == 0 )); then
    git merge --ff-only main >/dev/null 2>&1 || true
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
      git -c credential.helper= push origin "$WORK_BRANCH" >/dev/null 2>&1 || true
  fi

  rm -f "$askpass"
  unset LAB3_GITHUB_TOKEN
  return "$status"
}

declare -a SUITE_RESULTS=()

for variant in "${VARIANTS[@]}"; do
  echo
  echo "===== START $variant $(date --iso-8601=seconds) ====="
  "$ROOT/run_variant.sh" "$variant"
  run_status=$?

  archive=$(archive_variant "$variant" "$run_status")
  archive_status=$?
  if (( archive_status != 0 )); then
    echo "ARCHIVE_FAILED[$variant]=$archive_status"
    SUITE_RESULTS+=("$variant:run=$run_status,archive=$archive_status,publish=skipped")
    continue
  fi
  echo "ARCHIVE_COMPLETE[$variant]=$archive"

  publish_and_merge "$variant" "$archive"
  publish_status=$?
  SUITE_RESULTS+=("$variant:run=$run_status,archive=0,publish_merge=$publish_status")
  echo "===== END $variant $(date --iso-8601=seconds) ====="
done

"$ENV/bin/python" "$ROOT/compare_results.py" "$RESULTS_ROOT" \
  2>&1 | tee "$RESULTS_ROOT/final_remaining_attention_comparison.log"
COMPARE_STATUS=${PIPESTATUS[0]}

echo
echo "===== REMAINING ATTENTION SUITE SUMMARY ====="
printf '%s\n' "${SUITE_RESULTS[@]}"
echo "comparison_status=$COMPARE_STATUS"
echo "suite_log=$SUITE_LOG"
echo "REMAINING_ATTENTION_SUITE_FINISHED=$(date --iso-8601=seconds)"
echo "Terminal remains open after this script returns."
unset TOKEN

