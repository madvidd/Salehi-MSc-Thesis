#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

EXPERIMENT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${BASE:-/home/server01/M}
CODE="$EXPERIMENT_ROOT/Code"
RESULTS_ROOT=$(tr -d '\r\n' < "$BASE/Results/LATEST_SEAM_AV2_20EPOCH_4TEST.txt" 2>/dev/null)
DATA_ROOT=$(tr -d '\r\n' < "$RESULTS_ROOT/DATA_ROOT.txt" 2>/dev/null)
ENV_DIR="$BASE/Codes/envs/seam_av2_mamba_torch211"
PYTHON="$ENV_DIR/bin/python"
GPU_IDS=${SEAM_GPU_IDS:-0,1}
VARIANTS=(baseline uncertainty_target_context relative_geometry_bias qknorm)
STATUS=0

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_MODULE_LOADING=LAZY
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256,garbage_collection_threshold:0.8}

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: verified SEAM environment is missing: $ENV_DIR"
  exit 1
fi
if [ ! -d "$CODE" ] || [ ! -s "$CODE/train.py" ]; then
  echo "ERROR: generated SEAM source is incomplete: $CODE"
  exit 1
fi
if [ ! -d "$DATA_ROOT/train" ] || [ ! -d "$DATA_ROOT/val" ]; then
  echo "ERROR: processed AV2 train/val data is missing: $DATA_ROOT"
  exit 1
fi

exec 9>"$RESULTS_ROOT/.suite.lock"
if ! flock -n 9; then
  echo "ERROR: this exact SEAM four-test suite is already running."
  exit 1
fi

echo "SEAM_AV2_20EPOCH_4TEST_START=$(date --iso-8601=seconds)"
echo "Experiment: $EXPERIMENT_ROOT"
echo "Results:    $RESULTS_ROOT"
echo "Data:       $DATA_ROOT"
echo "Variants:   ${VARIANTS[*]}"
echo "Control: 20 epochs, effective global batch 32, AdamW, LR 1e-3 to 1e-5"
echo "Hardware: GPUs $GPU_IDS, batch 8/GPU, accumulation 2"

if [ ! -f "$RESULTS_ROOT/PREFLIGHT_COMPLETE" ]; then
  mkdir -p "$RESULTS_ROOT/preflight"
  echo "Running structural, optimiser, CUDA, and warning-hardening audits..."
  "$PYTHON" "$EXPERIMENT_ROOT/preflight_seam_20epoch_4test.py" \
    --code "$CODE" --gpus "$GPU_IDS" \
    2>&1 | tee "$RESULTS_ROOT/preflight/structural_cuda.log"
  STATUS=${PIPESTATUS[0]}

  if [ "$STATUS" -eq 0 ]; then
    echo "Running one real AV2 streamed-batch forward/backward test per variant..."
    "$PYTHON" "$EXPERIMENT_ROOT/real_data_smoke_test.py" \
      --code "$CODE" --data-root "$DATA_ROOT" \
      2>&1 | tee "$RESULTS_ROOT/preflight/real_data_cuda.log"
    STATUS=${PIPESTATUS[0]}
  fi

  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: preflight failed; no long training run was started."
    exit "$STATUS"
  fi

  if grep -aEi 'Traceback|CUDA error|out of memory|NCCL WARN|RuntimeError|UserWarning|FutureWarning|DeprecationWarning' \
      "$RESULTS_ROOT/preflight/structural_cuda.log" \
      "$RESULTS_ROOT/preflight/real_data_cuda.log" >/dev/null 2>&1; then
    echo "ERROR: preflight produced a warning or error signature."
    exit 1
  fi
  date --iso-8601=seconds > "$RESULTS_ROOT/PREFLIGHT_COMPLETE"
  echo "ALL_PREFLIGHT_CHECKS_PASSED=True"
else
  echo "REUSING_PASSED_PREFLIGHT=True"
fi

finalize_stale_attempt() {
  local run=$1
  local start_file="$run/CURRENT_ATTEMPT_START_SECONDS"
  local start last_log active delta
  [ -s "$start_file" ] || return 0
  start=$(tr -dc '0-9' < "$start_file")
  last_log=$(stat -c %Y "$run/train.log" 2>/dev/null)
  active=$(tr -dc '0-9' < "$run/ACTIVE_SECONDS.txt" 2>/dev/null)
  [ -n "$active" ] || active=0
  if [ -n "$start" ] && [ -n "$last_log" ] && [ "$last_log" -gt "$start" ]; then
    delta=$((last_log - start))
    active=$((active + delta))
    printf '%s\n' "$active" > "$run/ACTIVE_SECONDS.txt"
    printf '%s\t%s\t%s\t%s\n' "$start" "$last_log" "$delta" "recovered_after_interruption" \
      >> "$run/ATTEMPTS.tsv"
  fi
  rm -f "$start_file"
}

record_attempt_end() {
  local run=$1 start=$2 outcome=$3 now active delta
  now=$(date +%s)
  active=$(tr -dc '0-9' < "$run/ACTIVE_SECONDS.txt" 2>/dev/null)
  [ -n "$active" ] || active=0
  delta=$((now - start))
  [ "$delta" -lt 0 ] && delta=0
  printf '%s\n' "$((active + delta))" > "$run/ACTIVE_SECONDS.txt"
  printf '%s\t%s\t%s\t%s\n' "$start" "$now" "$delta" "$outcome" >> "$run/ATTEMPTS.tsv"
  rm -f "$run/CURRENT_ATTEMPT_START_SECONDS"
}

cd "$CODE" || exit 1

for INDEX in 0 1 2 3; do
  VARIANT=${VARIANTS[$INDEX]}
  NUMBER=$(printf '%02d' $((INDEX + 1)))
  RUN="$RESULTS_ROOT/${NUMBER}_${VARIANT}"
  LOG="$RUN/train.log"
  mkdir -p "$RUN"
  finalize_stale_attempt "$RUN"

  if [ -f "$RUN/TRAINING_COMPLETE" ]; then
    echo "SKIP_COMPLETE=$VARIANT"
    "$PYTHON" "$EXPERIMENT_ROOT/generate_seam_20epoch_summary.py" \
      --results-root "$RESULTS_ROOT"
    if [ ! -f "$RUN/PUBLISHED_TO_GITHUB" ]; then
      bash "$EXPERIMENT_ROOT/publish_seam_20epoch_snapshot.sh" \
        "$EXPERIMENT_ROOT" "$RESULTS_ROOT" "${VARIANT}_complete"
      STATUS=$?
      if [ "$STATUS" -eq 0 ]; then
        date --iso-8601=seconds > "$RUN/PUBLISHED_TO_GITHUB"
      else
        echo "ERROR: $VARIANT is locally complete, but publication failed."
        break
      fi
    fi
    continue
  fi

  CKPT=null
  if [ -s "$RUN/checkpoints/last.ckpt" ]; then
    CKPT="$RUN/checkpoints/last.ckpt"
    echo "RESUME_VARIANT=$VARIANT checkpoint=$CKPT"
  else
    echo "START_VARIANT=$VARIANT"
  fi

  {
    echo "variant=$VARIANT"
    echo "epochs=20"
    echo "seed=2333"
    echo "optimizer=AdamW"
    echo "peak_lr=0.001"
    echo "min_lr=0.00001"
    echo "warmup_ratio=0.167"
    echo "weight_decay=0.01"
    echo "gradient_clip_norm=5"
    echo "effective_global_batch=32"
    echo "per_gpu_microbatch=8"
    echo "gradient_accumulation=2"
    echo "world_size=2"
    echo "precision=32"
    echo "data_root=$DATA_ROOT"
    echo "resume_checkpoint=$CKPT"
  } > "$RUN/CONTROLLED_CONFIGURATION.txt"

  ATTEMPT_START=$(date +%s)
  printf '%s\n' "$ATTEMPT_START" > "$RUN/CURRENT_ATTEMPT_START_SECONDS"
  mkdir -p "$RUN/attempt_logs"
  ATTEMPT_LOG="$RUN/attempt_logs/attempt_$(date +%Y%m%d-%H%M%S).log"
  echo "ATTEMPT_START=$VARIANT $(date --iso-8601=seconds)"

  "$PYTHON" train.py \
    seed=2333 \
    gpus=2 \
    epochs=20 \
    batch_size=8 \
    checkpoint="$CKPT" \
    output_dir="$RUN" \
    callbacks.0.save_top_k=-1 \
    callbacks.0.save_last=true \
    callbacks.0.every_n_epochs=1 \
    datamodule.pl_module.data_root="$DATA_ROOT" \
    datamodule.pl_module.train_batch_size=8 \
    datamodule.pl_module.test_batch_size=16 \
    datamodule.pl_module.num_workers=4 \
    model.pl_module.model.variant="$VARIANT" \
    model.pl_module.optim.lr=0.001 \
    model.pl_module.optim.min_lr=0.00001 \
    model.pl_module.optim.weight_decay=0.01 \
    model.pl_module.optim.warmup_ratio=0.167 \
    trainer.devices=2 \
    trainer.accumulate_grad_batches=2 \
    trainer.strategy=ddp_find_unused_parameters_false \
    trainer.sync_batchnorm=true \
    trainer.gradient_clip_val=5 \
    trainer.gradient_clip_algorithm=norm \
    2>&1 | tee -a "$LOG" "$ATTEMPT_LOG"
  TRAIN_STATUS=${PIPESTATUS[0]}
  record_attempt_end "$RUN" "$ATTEMPT_START" "status_$TRAIN_STATUS"

  "$PYTHON" "$EXPERIMENT_ROOT/generate_seam_20epoch_summary.py" \
    --results-root "$RESULTS_ROOT"

  if [ "$TRAIN_STATUS" -ne 0 ] || [ ! -f "$RUN/TRAINING_COMPLETE" ]; then
    echo "ERROR: $VARIANT stopped before completion (status $TRAIN_STATUS)."
    echo "Rerun the same launch command to resume from last.ckpt."
    STATUS=$TRAIN_STATUS
    [ "$STATUS" -eq 0 ] && STATUS=1
    break
  fi

  if grep -aEi 'Traceback|CUDA error|out of memory|NCCL WARN|Error executing job|RuntimeError|AssertionError|FATAL' \
      "$ATTEMPT_LOG" >/dev/null 2>&1; then
    echo "ERROR: a fatal diagnostic signature exists in the latest $VARIANT attempt."
    STATUS=1
    break
  fi

  bash "$EXPERIMENT_ROOT/publish_seam_20epoch_snapshot.sh" \
    "$EXPERIMENT_ROOT" "$RESULTS_ROOT" "${VARIANT}_complete"
  STATUS=$?
  if [ "$STATUS" -ne 0 ]; then
    echo "ERROR: $VARIANT completed locally, but publication failed."
    echo "The next variant was not started; relaunch safely retries publication."
    break
  fi
  date --iso-8601=seconds > "$RUN/PUBLISHED_TO_GITHUB"
  echo "VARIANT_COMPLETE_AND_PUBLISHED=$VARIANT"
done

if [ "$STATUS" -eq 0 ]; then
  COMPLETE_COUNT=0
  for INDEX in 0 1 2 3; do
    NUMBER=$(printf '%02d' $((INDEX + 1)))
    [ -f "$RESULTS_ROOT/${NUMBER}_${VARIANTS[$INDEX]}/TRAINING_COMPLETE" ] && \
      COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
  done
  if [ "$COMPLETE_COUNT" -eq 4 ]; then
    date --iso-8601=seconds > "$RESULTS_ROOT/SUITE_COMPLETE"
    "$PYTHON" "$EXPERIMENT_ROOT/generate_seam_20epoch_summary.py" \
      --results-root "$RESULTS_ROOT"
    bash "$EXPERIMENT_ROOT/publish_seam_20epoch_snapshot.sh" \
      "$EXPERIMENT_ROOT" "$RESULTS_ROOT" "final_results"
    STATUS=$?
    if [ "$STATUS" -eq 0 ]; then
      date --iso-8601=seconds > "$RESULTS_ROOT/FINAL_PUBLISHED_TO_GITHUB"
      echo "SEAM_AV2_20EPOCH_4TEST_COMPLETE=True"
    fi
  else
    STATUS=1
  fi
fi

echo "Suite status: $STATUS"
echo "Terminal remains open."
exit "$STATUS"
