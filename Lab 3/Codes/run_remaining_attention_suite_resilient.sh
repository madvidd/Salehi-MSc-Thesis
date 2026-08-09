#!/usr/bin/env bash

# Run the three non-baseline Lab 3 attention variants sequentially. Every
# attempt is checkpoint-aware. Every result is archived locally and its small
# reproducibility files are merged directly into main before the suite advances.

set -uo pipefail

BASE=/home/server01/M
TOKEN_FILE="$BASE/Token/Token.txt"
ROOT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")
RESULTS_ROOT=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
ENV_POINTER="$BASE/Codes/LATEST_SHARP_ATTENTION_ENV.txt"
ENV=$(tr -d '\r\n' < "$ENV_POINTER" 2>/dev/null)
CONDA="$BASE/Codes/AV2/miniforge3/bin/conda"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PREPARER="$SCRIPT_DIR/prepare_remaining_attention_runtime.py"
RUNNER_TEMPLATE="$SCRIPT_DIR/run_variant_resilient.sh"
GIT=/usr/bin/git
ASKPASS="$BASE/Token/git-token-askpass.sh"
if [[ -n "${LAB3_VARIANTS:-}" ]]; then
  read -r -a VARIANTS <<< "$LAB3_VARIANTS"
else
  VARIANTS=(qknorm talking_heads qknorm_talking_heads)
fi

for variant in "${VARIANTS[@]}"; do
  case "$variant" in
    qknorm|talking_heads|qknorm_talking_heads) ;;
    *)
      echo "FATAL: unsupported LAB3_VARIANTS entry: $variant"
      exit 2
      ;;
  esac
done
MAX_ATTEMPTS=${LAB3_MAX_ATTEMPTS:-4}
SUITE_STAMP=$(date +%Y%m%d-%H%M%S)
SUITE_LOG="$RESULTS_ROOT/remaining_attention_suite_$SUITE_STAMP.log"

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$SUITE_LOG") 2>&1

echo "REMAINING_ATTENTION_SUITE_START=$(date --iso-8601=seconds)"
echo "ROOT=$ROOT"
echo "RESULTS_ROOT=$RESULTS_ROOT"
echo "ENV=$ENV"
echo "SUITE_LOG=$SUITE_LOG"
echo "VARIANTS=${VARIANTS[*]}"
echo "MAX_ATTEMPTS=$MAX_ATTEMPTS"
echo "BASELINE_MHA_IS_EXPLICITLY_EXCLUDED=True"
echo "TRAINING_CONFIGURATION_CHANGED=False"
echo "VALIDATION_LOG_BATCH_SIZE_EXPLICIT=True"
echo "PUBLICATION_TARGET=main"

for required in \
  "$TOKEN_FILE" \
  "$ROOT" \
  "$ENV_POINTER" \
  "$ENV/bin/python" \
  "$PREPARER" \
  "$RUNNER_TEMPLATE" \
  "$GIT"
do
  if [[ ! -e "$required" ]]; then
    echo "FATAL: required path is missing: $required"
    echo "Terminal remains open after this script returns."
    exit 1
  fi
done

install -m 755 "$RUNNER_TEMPLATE" "$ROOT/run_variant.sh"
"$ENV/bin/python" "$PREPARER"
PREPARE_STATUS=$?
if (( PREPARE_STATUS != 0 )); then
  echo "FATAL: runtime compatibility preparation failed: $PREPARE_STATUS"
  exit "$PREPARE_STATUS"
fi

RUNTIME_STATUS=$("$ENV/bin/python" - <<'PY'
import torch

ok = (
    torch.__version__.startswith("2.8.0")
    and torch.version.cuda == "12.6"
    and torch.cuda.is_available()
    and torch.cuda.device_count() >= 3
)
print("ok" if ok else "invalid")
PY
)
if [[ "$RUNTIME_STATUS" != ok ]]; then
  echo "FATAL: validated PyTorch 2.8 CUDA 12.6 environment is not active."
  exit 1
fi

chmod 600 "$TOKEN_FILE"
TOKEN=$("$ENV/bin/python" - "$TOKEN_FILE" <<'PY'
import pathlib
import re
import sys

data = pathlib.Path(sys.argv[1]).read_bytes()
match = re.search(rb"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", data)
if match:
    print(match.group(0).decode())
else:
    text = data.decode("utf-16", errors="ignore")
    match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
    print(match.group(0) if match else "")
PY
)
if [[ -z "$TOKEN" ]]; then
  echo "FATAL: no GitHub PAT was found in $TOKEN_FILE"
  exit 1
fi

USER_JSON=$(mktemp)
REPO_JSON=$(mktemp)
USER_HTTP=$(curl -sS -o "$USER_JSON" -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/user)
REPO_HTTP=$(curl -sS -o "$REPO_JSON" -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/madvidd/Thesis)
LOGIN=$("$ENV/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1])).get("login",""))' \
  "$USER_JSON" 2>/dev/null)
PUSH=$("$ENV/bin/python" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1])).get("permissions",{}).get("push",False)).lower())' \
  "$REPO_JSON" 2>/dev/null)
rm -f "$USER_JSON" "$REPO_JSON"

if [[ "$USER_HTTP" != 200 || "$REPO_HTTP" != 200 || \
      "$LOGIN" != madviddd || "$PUSH" != true ]]; then
  echo "FATAL: GitHub verification failed: user_http=$USER_HTTP repo_http=$REPO_HTTP account=$LOGIN push=$PUSH"
  unset TOKEN
  exit 1
fi
echo "GITHUB_AUTHENTICATED_ACCOUNT=$LOGIN"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'case "$1" in' \
  '  *Username*) printf "%s\n" "madviddd" ;;' \
  '  *Password*) tr -d "\r\n[:space:]" < /home/server01/M/Token/Token.txt ;;' \
  'esac' > "$ASKPASS"
chmod 700 "$ASKPASS"

unset GIT_TEMPLATE_DIR GIT_EXEC_PATH
export GIT_EXEC_PATH=$("$GIT" --exec-path)

archive_variant() {
  local variant=$1
  local run_status=$2
  local result_kind=$3
  local stamp archive out code file rel

  stamp=$(date +%Y%m%d-%H%M%S)
  archive="$BASE/Results/Archives/Lab3_SHARP_Attention_${variant}_${result_kind}_$stamp"
  out="$RESULTS_ROOT/$variant"
  code="$ROOT/variants/$variant/Code"
  mkdir -p "$archive/configuration" "$archive/reproducibility"

  {
    echo "variant=$variant"
    echo "result_kind=$result_kind"
    echo "saved=$(date --iso-8601=seconds)"
    echo "training_exit_code=$run_status"
    echo "code=$code"
    echo "results=$out"
    echo "environment=$ENV"
    echo "pytorch=$("$ENV/bin/python" -c 'import torch; print(torch.__version__)')"
    echo "cuda_runtime=$("$ENV/bin/python" -c 'import torch; print(torch.version.cuda)')"
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
    echo "result_kind=$result_kind"
    echo
    echo "===== METRICS ====="
    [[ -f "$out/metrics.json" ]] && cat "$out/metrics.json"
    echo
    echo "===== FINAL TRAIN LOG LINES ====="
    [[ -f "$out/train.log" ]] && tr '\r' '\n' < "$out/train.log" | tail -300
    echo
    echo "===== FINAL EVALUATION LOG LINES ====="
    [[ -f "$out/eval.log" ]] && tr '\r' '\n' < "$out/eval.log" | tail -300
    echo
    echo "===== DIAGNOSTICS ====="
    find "$out/diagnostics" -maxdepth 1 -type f -print -exec cat {} \; 2>/dev/null
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
    "$code/src/model/pl_modules.py" \
    "$ROOT/run_variant.sh" \
    "$ROOT/experiment_manifest.json"
  do
    [[ -f "$file" ]] && cp -p "$file" "$archive/reproducibility/$(basename "$file")"
  done

  "$ENV/bin/python" -m pip freeze > "$archive/pip-freeze.txt" 2>&1
  [[ -x "$CONDA" ]] && "$CONDA" list -p "$ENV" > "$archive/conda-list.txt" 2>&1
  nvidia-smi > "$archive/nvidia-smi.txt" 2>&1

  tar -czf "$archive/results_and_checkpoints.tar.gz" \
    -C "$RESULTS_ROOT" "$variant" || return 1
  tar -czf "$archive/code.tar.gz" \
    -C "$ROOT/variants/$variant" Code || return 1
  sha256sum "$archive"/*.tar.gz > "$archive/SHA256SUMS.txt" || return 1

  printf '%s\n' "$archive" \
    > "$BASE/Results/LATEST_LAB3_${variant^^}_ARCHIVE.txt"
  printf '%s\n' "$archive"
}

publish_archive() {
  local variant=$1
  local archive=$2
  local result_kind=$3
  local stamp clone_root rel dest file item status attempt local_sha remote_sha

  stamp=$(date +%Y%m%d-%H%M%S)
  clone_root="$BASE/Codes/Lab3_Publication_${variant}_${stamp}"
  if [[ "$result_kind" == completed ]]; then
    rel="Lab 3/Main_Results/Attention_Experiments/$(basename "$archive")"
  else
    rel="Lab 3/Main_Results/Attention_Experiments/Failed_Runs/$(basename "$archive")"
  fi

  GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    "$GIT" -c credential.helper= clone \
      --branch main --single-branch \
      https://github.com/madvidd/Thesis.git "$clone_root" || return 1

  dest="$clone_root/$rel"
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
    echo "Result kind: $result_kind"
    echo
    echo "Large archives, checkpoints, and full logs remain on Lab 3:"
    echo
    echo "\`$archive\`"
  } > "$dest/README.md"

  if grep -RIlE 'github_pat_|ghp_' "$dest" >/dev/null 2>&1; then
    echo "PUBLISH_ERROR[$variant]: credential text detected"
    return 1
  fi
  if find "$dest" -type f -size +10M | grep -q .; then
    echo "PUBLISH_ERROR[$variant]: file larger than 10 MB detected"
    return 1
  fi

  "$GIT" -C "$clone_root" config user.name "Seyed Mohammad Salehi"
  "$GIT" -C "$clone_root" config user.email "madviddd@users.noreply.github.com"
  "$GIT" -C "$clone_root" config credential.helper ""
  "$GIT" -C "$clone_root" config core.askPass "$ASKPASS"
  "$GIT" -C "$clone_root" config credential.username madviddd
  "$GIT" -C "$clone_root" config pull.rebase false
  "$GIT" -C "$clone_root" config merge.autoStash true
  "$GIT" -C "$clone_root" add -- "$rel"

  if "$GIT" -C "$clone_root" diff --cached --quiet -- "$rel"; then
    echo "PUBLISH_NO_CHANGES[$variant]"
  else
    "$GIT" -C "$clone_root" commit \
      -m "Add Lab 3 $variant $result_kind attention results" -- "$rel" \
      || return 1
  fi

  status=1
  for attempt in 1 2 3 4 5; do
    echo "PUBLISH_ATTEMPT[$variant]=$attempt/5"
    GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
      "$GIT" -C "$clone_root" -c credential.helper= \
      pull --no-rebase origin main || {
        sleep $(( attempt * 10 ))
        continue
      }
    GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
      "$GIT" -C "$clone_root" -c credential.helper= \
      push origin main && {
        status=0
        break
      }
    sleep $(( attempt * 10 ))
  done
  (( status == 0 )) || return "$status"

  local_sha=$("$GIT" -C "$clone_root" rev-parse HEAD)
  remote_sha=$(GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
    "$GIT" -C "$clone_root" -c credential.helper= \
    ls-remote origin refs/heads/main | awk '{print $1}')
  if [[ "$local_sha" != "$remote_sha" ]]; then
    echo "PUBLISH_ERROR[$variant]: remote main verification failed"
    return 1
  fi

  echo "PUBLISH_PULL_MERGE_PUSH_VERIFIED[$variant]=$remote_sha"
  echo "PUBLICATION_CLONE_PRESERVED[$variant]=$clone_root"
  return 0
}

wait_for_gpu_release() {
  local waited=0 pids
  while (( waited < 180 )); do
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | tr -d ' ' | sed '/^$/d')
    [[ -z "$pids" ]] && {
      echo "GPU_COMPUTE_PROCESSES_RELEASED=True"
      return 0
    }
    sleep 5
    waited=$(( waited + 5 ))
  done
  echo "GPU_RELEASE_TIMEOUT_AFTER_SECONDS=$waited"
  return 1
}

declare -a SUITE_RESULTS=()
SUITE_STATUS=0
PUBLICATION_BLOCKED=0

for variant in "${VARIANTS[@]}"; do
  echo
  echo "===== START $variant $(date --iso-8601=seconds) ====="
  run_status=1

  for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
    echo "RUN_ATTEMPT[$variant]=$attempt/$MAX_ATTEMPTS"
    LAB3_ATTEMPT_ID=$attempt "$ROOT/run_variant.sh" "$variant"
    run_status=$?

    if (( run_status == 0 )) && \
       [[ -f "$RESULTS_ROOT/$variant/COMPLETE" ]] && \
       [[ -s "$RESULTS_ROOT/$variant/metrics.json" ]]; then
      break
    fi

    echo "RUN_ATTEMPT_FAILED[$variant]=$run_status"
    if (( attempt < MAX_ATTEMPTS )); then
      if [[ -f "$RESULTS_ROOT/$variant/run/checkpoints/last.ckpt" ]]; then
        echo "RETRY_WILL_VALIDATE_AND_RESUME[$variant]=$RESULTS_ROOT/$variant/run/checkpoints/last.ckpt"
      else
        echo "RETRY_WILL_RESTART_WITHOUT_CHECKPOINT[$variant]=True"
      fi
      wait_for_gpu_release || true
      nvidia-smi
      echo "Retrying $variant after a 90-second cooldown."
      sleep 90
    fi
  done

  if (( run_status == 0 )) && \
     [[ -f "$RESULTS_ROOT/$variant/COMPLETE" ]] && \
     [[ -s "$RESULTS_ROOT/$variant/metrics.json" ]]; then
    result_kind=completed
  else
    result_kind=failed
    SUITE_STATUS=1
    echo "PERSISTENT_RUN_FAILURE[$variant]=$run_status"
  fi

  archive=$(archive_variant "$variant" "$run_status" "$result_kind")
  archive_status=$?
  if (( archive_status != 0 )); then
    echo "ARCHIVE_FAILED[$variant]=$archive_status"
    SUITE_RESULTS+=("$variant:run=$run_status,archive=$archive_status,publish=blocked")
    SUITE_STATUS=1
    break
  fi
  echo "ARCHIVE_COMPLETE[$variant]=$archive"

  publish_archive "$variant" "$archive" "$result_kind"
  publish_status=$?
  SUITE_RESULTS+=("$variant:run=$run_status,archive=0,publish_merge=$publish_status")
  if (( publish_status != 0 )); then
    echo "PUBLISH_OR_MERGE_FAILED[$variant]=$publish_status"
    SUITE_STATUS=1
    PUBLICATION_BLOCKED=1
    break
  fi

  echo "===== END $variant $(date --iso-8601=seconds) ====="
done

COMPARE_STATUS=skipped
if (( SUITE_STATUS == 0 )); then
  "$ENV/bin/python" "$ROOT/compare_results.py" "$RESULTS_ROOT" \
    2>&1 | tee "$RESULTS_ROOT/final_remaining_attention_comparison.log"
  COMPARE_STATUS=${PIPESTATUS[0]}
  (( COMPARE_STATUS == 0 )) || SUITE_STATUS=$COMPARE_STATUS
fi

echo
echo "===== REMAINING ATTENTION SUITE SUMMARY ====="
printf '%s\n' "${SUITE_RESULTS[@]}"
echo "comparison_status=$COMPARE_STATUS"
echo "publication_blocked=$PUBLICATION_BLOCKED"
echo "suite_status=$SUITE_STATUS"
echo "suite_log=$SUITE_LOG"
echo "REMAINING_ATTENTION_SUITE_FINISHED=$(date --iso-8601=seconds)"
echo "Terminal remains open after this script returns."
unset TOKEN
exit "$SUITE_STATUS"
