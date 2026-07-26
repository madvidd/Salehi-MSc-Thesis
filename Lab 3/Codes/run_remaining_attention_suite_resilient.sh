#!/usr/bin/env bash

# Run only the three remaining Lab 3 attention variants. A variant must train,
# evaluate, and validate its metrics before it can be published or the suite can
# advance. Native failures are retried once and retain diagnostics/checkpoints.

set -uo pipefail

BASE=/home/server01/M
TOKEN_FILE="$BASE/Token/Token.txt"
ROOT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")
RESULTS_ROOT=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
ENV_POINTER="$BASE/Codes/LATEST_SHARP_ATTENTION_ENV.txt"
ENV=
if [[ -s "$ENV_POINTER" ]]; then
  ENV=$(tr -d '\r\n' < "$ENV_POINTER")
fi
CONDA="$BASE/Codes/AV2/miniforge3/bin/conda"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
PREPARER="$SCRIPT_DIR/prepare_remaining_attention_runtime.py"
RUNNER_TEMPLATE="$SCRIPT_DIR/run_variant_resilient.sh"
WORK_BRANCH=lab3-sharp-attention-ablation
VARIANTS=(qknorm talking_heads qknorm_talking_heads)
MAX_ATTEMPTS=${LAB3_MAX_ATTEMPTS:-2}
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
echo "VALIDATION_LOG_BATCH_SIZE_EXPLICIT=True"

for required in \
  "$TOKEN_FILE" \
  "$ROOT" \
  "$ENV_POINTER" \
  "$PREPARER" \
  "$RUNNER_TEMPLATE"
do
  if [[ ! -e "$required" ]]; then
    echo "FATAL: required path is missing: $required"
    echo "Terminal remains open after this script returns."
    exit 1
  fi
done

if [[ -z "$ENV" || ! -x "$ENV/bin/python" ]]; then
  echo "FATAL: invalid environment pointer: $ENV_POINTER"
  exit 1
fi

install -m 755 "$RUNNER_TEMPLATE" "$ROOT/run_variant.sh"
"$ENV/bin/python" "$PREPARER"
PREPARE_STATUS=$?
if (( PREPARE_STATUS != 0 )); then
  echo "FATAL: runtime compatibility preparation failed: $PREPARE_STATUS"
  echo "Terminal remains open after this script returns."
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
  echo "FATAL: the validated PyTorch 2.8 CUDA 12.6 environment is not active."
  echo "Run prepare_lab3_cuda126_env.sh before restarting the suite."
  exit 1
fi

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
REPO_JSON=$(mktemp)
USER_HTTP=$(curl -sS -o "$USER_JSON" -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/user)
REPO_HTTP=$(curl -sS -o "$REPO_JSON" -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/madviddd/Thesis)
LOGIN=$("$ENV/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1])).get("login",""))' \
  "$USER_JSON" 2>/dev/null)
ACCESS=$("$ENV/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1])).get("full_name",""))' \
  "$REPO_JSON" 2>/dev/null)
PUSH=$("$ENV/bin/python" -c \
  'import json,sys; print(str(json.load(open(sys.argv[1])).get("permissions",{}).get("push",False)).lower())' \
  "$REPO_JSON" 2>/dev/null)
rm -f "$USER_JSON" "$REPO_JSON"

if [[ "$USER_HTTP" != 200 || "$REPO_HTTP" != 200 || \
      "$LOGIN" != madviddd || "$ACCESS" != madviddd/Thesis || \
      "$PUSH" != true ]]; then
  echo "FATAL: GitHub verification failed: user_http=$USER_HTTP repo_http=$REPO_HTTP account=$LOGIN repository=$ACCESS push=$PUSH"
  unset TOKEN
  exit 1
fi
echo "GITHUB_AUTHENTICATED_ACCOUNT=$LOGIN"

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
    echo "validation_log_batch_size=explicit_local_scenario_count"
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
    [[ -f "$out/train.log" ]] && tr '\r' '\n' < "$out/train.log" | tail -250
    echo
    echo "===== FINAL EVALUATION LOG LINES ====="
    [[ -f "$out/eval.log" ]] && tr '\r' '\n' < "$out/eval.log" | tail -250
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

publish_completed() {
  local variant=$1
  local archive=$2
  local stamp rel dest askpass status file item pr_number

  if [[ ! -f "$archive/COMPLETE" || ! -s "$archive/FINAL_METRICS.json" ]]; then
    echo "PUBLISH_ERROR[$variant]: only completed, evaluated runs can be published"
    return 1
  fi
  if ! command -v gh >/dev/null 2>&1; then
    echo "PUBLISH_ERROR[$variant]: gh is required for the merge step"
    return 1
  fi

  stamp=$(basename "$archive")
  rel="Lab 3/Main_Results/Attention_Experiments/$stamp"
  dest="$REPO/$rel"
  askpass=$(mktemp)
  status=0

  export LAB3_GITHUB_TOKEN="$TOKEN"
  printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    '  *Username*) printf "%s\n" "madviddd" ;;' \
    '  *Password*) printf "%s\n" "$LAB3_GITHUB_TOKEN" ;;' \
    'esac' > "$askpass"
  chmod 700 "$askpass"

  cd "$REPO" || return 1
  git remote set-url origin https://github.com/madviddd/Thesis.git
  GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
    git -c credential.helper= fetch --prune origin \
    "+refs/heads/$WORK_BRANCH:refs/remotes/origin/$WORK_BRANCH" \
    "+refs/heads/main:refs/remotes/origin/main" || status=$?

  if (( status == 0 )); then
    git switch "$WORK_BRANCH" || status=$?
  fi
  if (( status == 0 )); then
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
      git -c credential.helper= pull --rebase origin "$WORK_BRANCH" || status=$?
  fi
  if (( status == 0 )); then
    git merge --no-edit origin/main || status=$?
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
    git config user.email "madviddd@users.noreply.github.com"
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
    pr_number=$(GH_TOKEN="$TOKEN" gh pr list \
      --repo madviddd/Thesis \
      --base main \
      --head "$WORK_BRANCH" \
      --state open \
      --json number \
      --jq '.[0].number // empty')
    if [[ -z "$pr_number" ]]; then
      GH_TOKEN="$TOKEN" gh pr create \
        --repo madviddd/Thesis \
        --base main \
        --head "$WORK_BRANCH" \
        --title "Publish Lab 3 $variant attention results" \
        --body "Automated publication of the completed and validated $variant run." \
        >/dev/null || status=$?
      if (( status == 0 )); then
        pr_number=$(GH_TOKEN="$TOKEN" gh pr list \
          --repo madviddd/Thesis \
          --base main \
          --head "$WORK_BRANCH" \
          --state open \
          --json number \
          --jq '.[0].number // empty')
      fi
    fi
  fi
  if (( status == 0 )); then
    if [[ -z "$pr_number" ]]; then
      echo "PUBLISH_ERROR[$variant]: pull request number was not found"
      status=1
    else
      GH_TOKEN="$TOKEN" gh pr merge "$pr_number" \
        --repo madviddd/Thesis \
        --merge \
        --delete-branch=false || status=$?
    fi
  fi

  if (( status == 0 )); then
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
      git -c credential.helper= fetch origin \
      "+refs/heads/main:refs/remotes/origin/main" || status=$?
  fi
  if (( status == 0 )); then
    git merge --ff-only origin/main || status=$?
  fi
  if (( status == 0 )); then
    GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
      git -c credential.helper= push origin "$WORK_BRANCH" || status=$?
  fi

  if (( status != 0 )); then
    git merge --abort 2>/dev/null || true
    git rebase --abort 2>/dev/null || true
    echo "PUBLISH_OR_MERGE_FAILED[$variant]=$status"
  else
    echo "PUBLISH_AND_MERGE_COMPLETE[$variant]"
  fi

  rm -f "$askpass"
  unset LAB3_GITHUB_TOKEN
  return "$status"
}

preserve_failed_attempt() {
  local variant=$1
  local attempt=$2
  local source="$RESULTS_ROOT/$variant"
  local destination

  [[ -d "$source" ]] || return 0
  destination="$RESULTS_ROOT/preserved_partial_runs/${variant}_attempt_${attempt}_$(
    date +%Y%m%d-%H%M%S
  )"
  mkdir -p "$(dirname "$destination")"
  mv "$source" "$destination"
  echo "PRESERVED_FAILED_ATTEMPT[$variant]=$destination"
}

declare -a SUITE_RESULTS=()
SUITE_STATUS=0

for variant in "${VARIANTS[@]}"; do
  echo
  echo "===== START $variant $(date --iso-8601=seconds) ====="
  run_status=1

  for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
    echo "RUN_ATTEMPT[$variant]=$attempt/$MAX_ATTEMPTS"
    "$ROOT/run_variant.sh" "$variant"
    run_status=$?

    if (( run_status == 0 )) && \
       [[ -f "$RESULTS_ROOT/$variant/COMPLETE" ]] && \
       [[ -s "$RESULTS_ROOT/$variant/metrics.json" ]]; then
      break
    fi

    echo "RUN_ATTEMPT_FAILED[$variant]=$run_status"
    if (( attempt < MAX_ATTEMPTS )); then
      if [[ -f "$RESULTS_ROOT/$variant/run/checkpoints/last.ckpt" ]]; then
        echo "RETRY_WILL_RESUME[$variant]=$RESULTS_ROOT/$variant/run/checkpoints/last.ckpt"
      else
        preserve_failed_attempt "$variant" "$attempt"
      fi
      nvidia-smi
      echo "Retrying $variant after a 60-second cooldown."
      sleep 60
    fi
  done

  if (( run_status != 0 )) || \
     [[ ! -f "$RESULTS_ROOT/$variant/COMPLETE" ]] || \
     [[ ! -s "$RESULTS_ROOT/$variant/metrics.json" ]]; then
    archive=$(archive_variant "$variant" "$run_status" failed)
    archive_status=$?
    echo "PERSISTENT_RUN_FAILURE[$variant]=$run_status"
    echo "FAILED_ARCHIVE[$variant]=$archive"
    SUITE_RESULTS+=("$variant:run=$run_status,archive=$archive_status,publish=blocked")
    SUITE_STATUS=1
    break
  fi

  archive=$(archive_variant "$variant" 0 completed)
  archive_status=$?
  if (( archive_status != 0 )); then
    echo "ARCHIVE_FAILED[$variant]=$archive_status"
    SUITE_RESULTS+=("$variant:run=0,archive=$archive_status,publish=blocked")
    SUITE_STATUS=1
    break
  fi
  echo "ARCHIVE_COMPLETE[$variant]=$archive"

  publish_completed "$variant" "$archive"
  publish_status=$?
  SUITE_RESULTS+=("$variant:run=0,archive=0,publish_merge=$publish_status")
  if (( publish_status != 0 )); then
    SUITE_STATUS=1
    break
  fi
  echo "===== END $variant $(date --iso-8601=seconds) ====="
done

COMPARE_STATUS=skipped
if (( SUITE_STATUS == 0 )); then
  "$ENV/bin/python" "$ROOT/compare_results.py" "$RESULTS_ROOT" \
    2>&1 | tee "$RESULTS_ROOT/final_remaining_attention_comparison.log"
  COMPARE_STATUS=${PIPESTATUS[0]}
  if (( COMPARE_STATUS != 0 )); then
    SUITE_STATUS=$COMPARE_STATUS
  fi
fi

echo
echo "===== REMAINING ATTENTION SUITE SUMMARY ====="
printf '%s\n' "${SUITE_RESULTS[@]}"
echo "comparison_status=$COMPARE_STATUS"
echo "suite_status=$SUITE_STATUS"
echo "suite_log=$SUITE_LOG"
echo "REMAINING_ATTENTION_SUITE_FINISHED=$(date --iso-8601=seconds)"
echo "Terminal remains open after this script returns."
unset TOKEN
exit "$SUITE_STATUS"
