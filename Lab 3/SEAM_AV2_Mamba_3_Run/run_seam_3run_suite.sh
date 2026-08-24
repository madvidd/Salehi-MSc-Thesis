#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

EXPERIMENT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE=${BASE:-/home/server01/M}
CODE="$EXPERIMENT_ROOT/Code"
RESULTS_ROOT=$(tr -d '\r\n' < "$BASE/Results/LATEST_SEAM_AV2_MAMBA_3RUN.txt")
DATA_ROOT=$(tr -d '\r\n' < "$RESULTS_ROOT/DATA_ROOT.txt")
ENV_DIR="$BASE/Codes/envs/seam_av2_mamba_torch211"
PYTHON="$ENV_DIR/bin/python"
VARIANTS=(baseline mamba_agent_add mamba_future_replace)
GPU_IDS=${SEAM_GPU_IDS:-0,1}
STATUS=0

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: SEAM environment is missing: $ENV_DIR"
  exit 1
fi
if [ ! -d "$DATA_ROOT/train" ] || [ ! -d "$DATA_ROOT/val" ]; then
  echo "ERROR: processed AV2 train/val data is missing: $DATA_ROOT"
  exit 1
fi

echo "SEAM_AV2_MAMBA_3RUN_START=$(date --iso-8601=seconds)"
echo "Experiment: $EXPERIMENT_ROOT"
echo "Results:    $RESULTS_ROOT"
echo "Data:       $DATA_ROOT"
echo "Variants:   ${VARIANTS[*]}"
echo "Control: 80 epochs, effective global batch 32, AdamW, LR 1e-3 to 1e-5"
echo "Hardware adaptation: GPUs $GPU_IDS, batch 8/GPU, accumulation 2"

"$PYTHON" "$EXPERIMENT_ROOT/preflight_seam_variants.py" \
  --code "$CODE" --gpus "$GPU_IDS"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
  echo "ERROR: all-variant CUDA preflight failed; no training was started."
  exit "$STATUS"
fi

cd "$CODE" || exit 1

for VARIANT in "${VARIANTS[@]}"; do
  RUN="$RESULTS_ROOT/$VARIANT"
  LOG="$RUN/train.log"
  mkdir -p "$RUN"

  if [ -f "$RUN/TRAINING_COMPLETE" ]; then
    echo "SKIP_COMPLETE=$VARIANT"
    "$PYTHON" "$EXPERIMENT_ROOT/generate_seam_summary.py" \
      --variant-dir "$RUN" --experiment "$EXPERIMENT_ROOT" \
      --variant "$VARIANT"
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
    echo "started=$(date --iso-8601=seconds)"
    echo "epochs=80"
    echo "seed=2333"
    echo "optimizer=AdamW"
    echo "peak_lr=0.001"
    echo "min_lr=0.00001"
    echo "warmup_ratio=0.167"
    echo "warmup_epochs=13"
    echo "weight_decay=0.01"
    echo "gradient_clip_norm=5"
    echo "effective_global_batch=32"
    echo "per_gpu_microbatch=8"
    echo "gradient_accumulation=2"
    echo "world_size=2"
    echo "precision=32"
    echo "data_root=$DATA_ROOT"
    echo "checkpoint=$CKPT"
  } > "$RUN/CONTROLLED_CONFIGURATION.txt"

  "$PYTHON" train.py \
    seed=2333 \
    gpus=2 \
    epochs=80 \
    batch_size=8 \
    checkpoint="$CKPT" \
    output_dir="$RUN" \
    datamodule.pl_module.data_root="$DATA_ROOT" \
    datamodule.pl_module.train_batch_size=8 \
    datamodule.pl_module.test_batch_size=16 \
    datamodule.pl_module.num_workers=4 \
    model.pl_module.model.variant="$VARIANT" \
    model.pl_module.model.mamba_d_state=16 \
    model.pl_module.model.mamba_d_conv=4 \
    model.pl_module.model.mamba_expand=2 \
    model.pl_module.model.mamba_future_depth=2 \
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
    2>&1 | tee -a "$LOG"
  TRAIN_STATUS=${PIPESTATUS[0]}

  "$PYTHON" "$EXPERIMENT_ROOT/generate_seam_summary.py" \
    --variant-dir "$RUN" --experiment "$EXPERIMENT_ROOT" \
    --variant "$VARIANT"

  bash "$EXPERIMENT_ROOT/publish_seam_result.sh" \
    "$VARIANT" "$RUN" "$RESULTS_ROOT" "$EXPERIMENT_ROOT"
  PUBLISH_STATUS=$?

  if [ "$TRAIN_STATUS" -ne 0 ] || [ ! -f "$RUN/TRAINING_COMPLETE" ]; then
    echo "ERROR: $VARIANT stopped before completion (status $TRAIN_STATUS)."
    echo "Rerun the same launch command to resume from last.ckpt."
    STATUS=$TRAIN_STATUS
    [ "$STATUS" -eq 0 ] && STATUS=1
    break
  fi

  echo "VARIANT_COMPLETE=$VARIANT"
  if [ "$PUBLISH_STATUS" -ne 0 ]; then
    echo "WARNING: $VARIANT completed locally; GitHub publication can be retried."
  fi
done

if [ "$STATUS" -eq 0 ]; then
  COMPLETE_COUNT=0
  for VARIANT in "${VARIANTS[@]}"; do
    [ -f "$RESULTS_ROOT/$VARIANT/TRAINING_COMPLETE" ] && \
      COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
  done
  if [ "$COMPLETE_COUNT" -eq 3 ]; then
    date --iso-8601=seconds > "$RESULTS_ROOT/SUITE_COMPLETE"
    echo "SEAM_AV2_MAMBA_3RUN_COMPLETE=True"
  else
    STATUS=1
  fi
fi

echo "Suite status: $STATUS"
echo "Terminal remains open."
exit "$STATUS"
