#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/suite.env"

PYTHON_BIN="$BASE/Codes/envs/sharp/bin/python"
DATASET="$BASE/Datasets/AV2/sharp_processed"
SUITE_LOG="$RESULTS_ROOT/suite.log"
LOCK_FILE="$RESULTS_ROOT/suite.lock"
MAX_ATTEMPTS=3
STATUS=1

SLUGS=(
  01_official_sharp_baseline
  02_qknorm_uncertainty_geometry
  03_qknorm_uncertainty_geometry_temporal_mamba
)
VARIANTS=(
  official_sharp_baseline
  qknorm_uncertainty_geometry
  qknorm_uncertainty_geometry_temporal_mamba
)

mkdir -p "$RESULTS_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: this final three-run suite is already active."
  echo "Lock: $LOCK_FILE"
  exit 1
fi

exec > >(tee -a "$SUITE_LOG") 2>&1

on_interrupt() {
  printf 'interrupted=%s\n' "$(date --iso-8601=seconds)" \
    > "$RESULTS_ROOT/SUITE_INTERRUPTED"
  echo "The suite was interrupted. Checkpoints and previous results are preserved."
  echo "Run the same launch command again to resume."
  exit 130
}
trap on_interrupt INT TERM HUP

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export NO_COLOR=1
export RICH_NO_COLOR=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=1048576
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
ulimit -n 65535 2>/dev/null || true

echo "SHARP_FINAL_3RUN_START=$(date --iso-8601=seconds)"
echo "Experiment root: $EXPERIMENT_ROOT"
echo "Results root:    $RESULTS_ROOT"
echo "Official commit: $OFFICIAL_COMMIT"
echo "Paper schedule: 80 epochs, 13 warm-up, global batch 32, LR 1e-4 -> 1e-5"
echo "Execution: four GPUs, batch 8/rank, four workers/rank, FP32, SyncBatchNorm"

if [ ! -x "$PYTHON_BIN" ] || [ ! -d "$DATASET/train" ] || [ ! -d "$DATASET/val" ]; then
  echo "ERROR: Python environment or AV2 processed dataset is missing."
  exit 1
fi

ORIGINAL_PYTHONPATH=${PYTHONPATH:-}

ENVIRONMENT_DIR="$RESULTS_ROOT/dissertation_artifacts/environment"
mkdir -p "$ENVIRONMENT_DIR"
cp -p "$EXPERIMENT_ROOT/SUITE_MANIFEST.json" \
  "$EXPERIMENT_ROOT/PINNED_SOURCE_SHA256.txt" "$ENVIRONMENT_DIR/"
uname -a > "$ENVIRONMENT_DIR/system.txt"
lscpu > "$ENVIRONMENT_DIR/lscpu.txt" 2>&1 || true
free -h > "$ENVIRONMENT_DIR/memory.txt" 2>&1 || true
nvidia-smi -q > "$ENVIRONMENT_DIR/nvidia-smi-q.txt" 2>&1 || true
"$PYTHON_BIN" -m pip freeze > "$ENVIRONMENT_DIR/pip-freeze.txt" 2>&1 || true

if ! nvidia-smi -L; then
  echo "ERROR: the compatible NVIDIA user-space runtime is not active."
  exit 1
fi

OTHER_TRAINING=$(ps -u "$USER" -o pid=,args= | \
  awk -v root="$EXPERIMENT_ROOT" '$0 ~ /[t]rain[.]py/ && index($0, root) == 0 {print}')
if [ -n "$OTHER_TRAINING" ]; then
  echo "ERROR: another training job is using Lab 2 resources:"
  echo "$OTHER_TRAINING"
  echo "No process was stopped."
  exit 1
fi

CURRENT_TRAINING=$(ps -u "$USER" -o pid=,args= | \
  awk -v root="$EXPERIMENT_ROOT" '$0 ~ /[t]rain[.]py/ && index($0, root) > 0 {print}')
if [ -n "$CURRENT_TRAINING" ]; then
  echo "ERROR: this same final suite still has active training ranks:"
  echo "$CURRENT_TRAINING"
  echo "No duplicate process was started and no process was stopped."
  exit 1
fi

TOKEN_CHECK=$(
  "$PYTHON_BIN" - "$BASE/Token/Token.txt" <<'PY'
import pathlib
import re
import sys

data = pathlib.Path(sys.argv[1]).read_bytes()
text = data.decode("utf-8-sig", "ignore") + "\n" + data.decode("utf-16", "ignore")
match = re.search(r"github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9]+", text)
print(match.group(0) if match else "")
PY
)
TOKEN_LOGIN=$(curl -fsSL -H "Authorization: Bearer $TOKEN_CHECK" \
  https://api.github.com/user 2>/dev/null | \
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("login", ""))' \
  2>/dev/null)
TOKEN_PUSH=$(curl -fsSL -H "Authorization: Bearer $TOKEN_CHECK" \
  https://api.github.com/repos/madvidd/Thesis 2>/dev/null | \
  "$PYTHON_BIN" -c \
  'import json,sys; print(str(json.load(sys.stdin).get("permissions", {}).get("push", False)).lower())' \
  2>/dev/null)
unset TOKEN_CHECK
if [ "$TOKEN_LOGIN" != "madviddd" ] || [ "$TOKEN_PUSH" != "true" ]; then
  echo "ERROR: the saved token cannot publish as madviddd to madvidd/Thesis."
  exit 1
fi
echo "GitHub publication preflight: madviddd with push permission"

if [ ! -f "$RESULTS_ROOT/PREFLIGHT_COMPLETE" ]; then
  echo "Running static, fused-CUDA, optimizer, four-GPU DDP, and real-batch checks..."
  "$PYTHON_BIN" "$EXPERIMENT_ROOT/preflight_final_suite.py" \
    --experiment "$EXPERIMENT_ROOT" \
    --results "$RESULTS_ROOT" \
    --python "$PYTHON_BIN" \
    --dataset "$DATASET" \
    --runtime-patch "$RUNTIME_PATCH" \
    --mamba-packages "$MAMBA_PACKAGES"
  STATUS=$?
  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: preflight failed; no 80-epoch training was started."
    exit "$STATUS"
  fi
else
  echo "Reusing completed final-suite preflight."
fi

if [ ! -s "$RESULTS_ROOT/NCCL_RUNTIME.env" ]; then
  echo "ERROR: validated NCCL runtime settings are missing."
  exit 1
fi
source "$RESULTS_ROOT/NCCL_RUNTIME.env"
echo "Validated NCCL runtime: NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}"

select_resume_checkpoint() {
  local checkpoint_dir="$1"
  "$PYTHON_BIN" - "$checkpoint_dir" <<'PY'
import pathlib
import sys
import torch

root = pathlib.Path(sys.argv[1])
candidates = sorted(root.glob("*.ckpt"), key=lambda item: item.stat().st_mtime, reverse=True)
valid = []
for path in candidates:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        valid.append((int(checkpoint.get("global_step", -1)), int(checkpoint.get("epoch", -1)), path))
    except Exception as error:
        print(f"IGNORING_INVALID_CHECKPOINT={path}: {error}", file=sys.stderr)
if valid:
    print(max(valid)[2])
PY
}

cleanup_current_variant() {
  local code_dir="$1"
  local run_dir="$2"
  local pids
  pids=$(
    "$PYTHON_BIN" - "$code_dir" "$run_dir" <<'PY'
import os
import pathlib
import sys

needles = tuple(sys.argv[1:])
for proc in pathlib.Path("/proc").glob("[0-9]*"):
    try:
        pid = int(proc.name)
        if pid == os.getpid() or pid == os.getppid():
            continue
        command = proc.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except (OSError, ValueError):
        continue
    if "train.py" in command and any(needle in command for needle in needles):
        print(pid)
PY
  )
  if [ -n "$pids" ]; then
    echo "Stopping only failed ranks from the current final-suite variant: $pids"
    kill -TERM $pids 2>/dev/null || true
    for _ in $(seq 1 20); do
      sleep 2
      local remaining=""
      for pid in $pids; do
        kill -0 "$pid" 2>/dev/null && remaining="$remaining $pid"
      done
      [ -z "$remaining" ] && break
    done
    for pid in $pids; do
      kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
    done
  fi
}

write_timing() {
  local attempt_file="$1"
  local output="$2"
  "$PYTHON_BIN" - "$attempt_file" "$output" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
duration = 0
attempts = 0
if source.is_file():
    for line in source.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 6:
            attempts += 1
            duration += int(fields[4])
payload = {"training_attempts": attempts, "training_duration_seconds": duration}
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

for INDEX in 0 1 2; do
  SLUG=${SLUGS[$INDEX]}
  VARIANT=${VARIANTS[$INDEX]}
  CODE_DIR="$EXPERIMENT_ROOT/variants/$SLUG/Code"
  VARIANT_ROOT="$RESULTS_ROOT/$SLUG"
  RUN_DIR="$VARIANT_ROOT/run"
  LOG="$VARIANT_ROOT/full_run.log"
  ATTEMPT_FILE="$VARIANT_ROOT/ATTEMPTS.tsv"
  ARTIFACTS="$VARIANT_ROOT/artifacts"
  mkdir -p "$RUN_DIR" "$ARTIFACTS"

  echo
  echo "===== FINAL SUITE RUN $((INDEX + 1))/3: $SLUG ====="

  if [ ! -f "$VARIANT_ROOT/COMPLETED" ]; then
    ATTEMPT=1
    TRAIN_STATUS=1
    while [ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]; do
      RESUME=$(select_resume_checkpoint "$RUN_DIR/checkpoints")
      START_SECONDS=$(date +%s)
      START_ISO=$(date --iso-8601=seconds)
      ATTEMPT_LOG="$VARIANT_ROOT/attempt_logs/attempt_${ATTEMPT}_$(date +%Y%m%d-%H%M%S).log"
      mkdir -p "$VARIANT_ROOT/attempt_logs"
      echo "TRAINING_ATTEMPT=$ATTEMPT START=$START_ISO RESUME=${RESUME:-none}"
      printf '\n===== ATTEMPT %s | %s | RESUME=%s =====\n' \
        "$ATTEMPT" "$START_ISO" "${RESUME:-none}" >> "$LOG"

      COMMAND=(
        "$PYTHON_BIN" train.py
        seed=2333
        gpus=4
        batch_size=8
        epochs=80
        "output_dir=$RUN_DIR"
        "datamodule.pl_module.data_root=$DATASET"
        datamodule.pl_module.num_workers=4
        model.pl_module.optim.lr=0.0001
        model.pl_module.optim.min_lr=0.00001
        model.pl_module.optim.warmup_ratio=0.1625
        model.pl_module.optim.weight_decay=0.01
        trainer.devices=4
        trainer.strategy=ddp_find_unused_parameters_false
        trainer.sync_batchnorm=true
        trainer.precision=32-true
      )
      if [ -n "$RESUME" ]; then
        COMMAND+=("checkpoint=$RESUME")
      fi

      printf '%q ' "${COMMAND[@]}" > "$VARIANT_ROOT/TRAIN_COMMAND.txt"
      printf '\n' >> "$VARIANT_ROOT/TRAIN_COMMAND.txt"

      cd "$CODE_DIR" || exit 1
      export SHARP_FINAL_VARIANT="$VARIANT"
      export PYTHONPATH="$RUNTIME_PATCH:$MAMBA_PACKAGES:$CODE_DIR:$ORIGINAL_PYTHONPATH"
      "${COMMAND[@]}" 2>&1 | tee -a "$LOG" "$ATTEMPT_LOG"
      TRAIN_STATUS=${PIPESTATUS[0]}
      END_SECONDS=$(date +%s)
      END_ISO=$(date --iso-8601=seconds)
      ELAPSED=$((END_SECONDS - START_SECONDS))
      if [ "$TRAIN_STATUS" -eq 0 ]; then
        if grep -aEi \
          'Traceback \(most recent call last\)|Error executing job with overrides|illegal memory access|CUDA out of memory|Trying to infer the `batch_size`|Support for mismatched key_padding_mask and attn_mask|NCCL WARN' \
          "$ATTEMPT_LOG" >/dev/null 2>&1; then
          echo "ERROR: a forbidden runtime warning/error was found in the successful attempt."
          grep -aEi \
            'Traceback \(most recent call last\)|Error executing job with overrides|illegal memory access|CUDA out of memory|Trying to infer the `batch_size`|Support for mismatched key_padding_mask and attn_mask|NCCL WARN' \
            "$ATTEMPT_LOG" | tail -100
          TRAIN_STATUS=1
        fi
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ATTEMPT" "$START_ISO" "$END_ISO" "$VARIANT" "$ELAPSED" \
        "$TRAIN_STATUS" "${RESUME:-none}" >> "$ATTEMPT_FILE"
      if [ "$TRAIN_STATUS" -eq 0 ]; then
        break
      fi
      echo "Training attempt $ATTEMPT failed with status $TRAIN_STATUS."
      cleanup_current_variant "$CODE_DIR" "$RUN_DIR"
      ATTEMPT=$((ATTEMPT + 1))
      if [ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]; then
        echo "Retrying from the newest valid checkpoint in 30 seconds."
        sleep 30
      fi
    done

    if [ "$TRAIN_STATUS" -ne 0 ]; then
      echo "ERROR: $SLUG exhausted $MAX_ATTEMPTS attempts."
      echo "All checkpoints are preserved; rerun the launcher to resume."
      exit "$TRAIN_STATUS"
    fi

    BEST_FILE="$RUN_DIR/BEST_CHECKPOINT.txt"
    if [ ! -s "$BEST_FILE" ]; then
      echo "ERROR: training finished but BEST_CHECKPOINT.txt is missing."
      exit 1
    fi
    BEST=$(tr -d '\r\n' < "$BEST_FILE")
    if [ ! -s "$BEST" ]; then
      echo "ERROR: best checkpoint is missing: $BEST"
      exit 1
    fi

    echo "Running full single-GPU validation for unbiased final metrics..."
    SHARP_FINAL_VARIANT="$VARIANT" \
    PYTHONPATH="$RUNTIME_PATCH:$MAMBA_PACKAGES:$CODE_DIR:$ORIGINAL_PYTHONPATH" \
      "$PYTHON_BIN" "$EXPERIMENT_ROOT/eval_checkpoint.py" \
      --code "$CODE_DIR" \
      --checkpoint "$BEST" \
      --dataset "$DATASET" \
      --output "$ARTIFACTS/metrics.json" 2>&1 | tee "$VARIANT_ROOT/evaluation.log"
    EVAL_STATUS=${PIPESTATUS[0]}
    if [ "$EVAL_STATUS" -ne 0 ]; then
      echo "ERROR: final evaluation failed; training checkpoints remain intact."
      exit "$EVAL_STATUS"
    fi
    if grep -aEi \
      'Traceback \(most recent call last\)|illegal memory access|CUDA out of memory|Trying to infer the `batch_size`|Support for mismatched key_padding_mask and attn_mask|NCCL WARN' \
      "$VARIANT_ROOT/evaluation.log" >/dev/null 2>&1; then
      echo "ERROR: a forbidden runtime warning/error was found during final evaluation."
      grep -aEi \
        'Traceback \(most recent call last\)|illegal memory access|CUDA out of memory|Trying to infer the `batch_size`|Support for mismatched key_padding_mask and attn_mask|NCCL WARN' \
        "$VARIANT_ROOT/evaluation.log" | tail -100
      exit 1
    fi
    write_timing "$ATTEMPT_FILE" "$VARIANT_ROOT/RUN_TIMING.json"
    printf 'completed=%s\nbest_checkpoint=%s\n' \
      "$(date --iso-8601=seconds)" "$BEST" > "$VARIANT_ROOT/COMPLETED"
  else
    echo "Training and final evaluation already completed; reusing saved results."
  fi

  "$PYTHON_BIN" "$EXPERIMENT_ROOT/generate_final_artifacts.py" \
    --results "$RESULTS_ROOT" --slug "$SLUG" --comparison
  STATUS=$?
  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: dissertation artifact generation failed for $SLUG."
    exit "$STATUS"
  fi

  "$EXPERIMENT_ROOT/publish_final_artifacts.sh" --slug "$SLUG"
  STATUS=$?
  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: result publication failed; the next run was not started."
    echo "Rerun the launcher to retry publication without retraining completed runs."
    exit "$STATUS"
  fi
  echo "RUN_COMPLETE_AND_PUBLISHED=$SLUG"
done

printf 'completed=%s\n' "$(date --iso-8601=seconds)" > "$RESULTS_ROOT/SUITE_RUNS_COMPLETED"
"$PYTHON_BIN" "$EXPERIMENT_ROOT/generate_final_artifacts.py" \
  --results "$RESULTS_ROOT" --comparison
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "ERROR: final completed-suite comparison generation failed."
  exit "$STATUS"
fi
"$EXPERIMENT_ROOT/publish_final_artifacts.sh" \
  --slug 03_qknorm_uncertainty_geometry_temporal_mamba
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "ERROR: final completed-suite comparison publication failed."
  exit "$STATUS"
fi
printf 'completed=%s\n' "$(date --iso-8601=seconds)" > "$RESULTS_ROOT/SUITE_COMPLETED"

echo
echo "SHARP_FINAL_3RUN_SUITE_COMPLETE"
echo "Results: $RESULTS_ROOT"
echo "All previous experiments remain unchanged."
exit 0
