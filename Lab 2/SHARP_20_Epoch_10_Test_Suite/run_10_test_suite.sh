#!/usr/bin/env bash
set -Eeuo pipefail

BASE="@@BASE@@"
EXPERIMENT_ROOT="@@EXPERIMENT_ROOT@@"
RESULTS_ROOT="@@RESULTS_ROOT@@"
CODE_DIR="@@CODE_DIR@@"
MAMBA_PACKAGES="@@MAMBA_PACKAGES@@"
RUNTIME_PATCH="@@RUNTIME_PATCH@@"
PYTHON_BIN="$BASE/Codes/envs/sharp/bin/python"
DATA_ROOT="$BASE/Datasets/AV2/sharp_processed"
SUITE_LOG="$RESULTS_ROOT/suite.log"

mkdir -p "$RESULTS_ROOT"
exec > >(tee -a "$SUITE_LOG") 2>&1

exec 9>"$RESULTS_ROOT/suite.lock"
if ! flock -n 9; then
  echo "ERROR: this ten-test suite is already running."
  exit 1
fi

source "$BASE/Codes/miniforge3/etc/profile.d/conda.sh"
conda activate "$BASE/Codes/envs/sharp"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$RUNTIME_PATCH:$MAMBA_PACKAGES:$CODE_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export NO_COLOR=1
export RICH_NO_COLOR=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_USE_COMM_NONBLOCKING=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ulimit -n 65535 2>/dev/null || true

test -x "$PYTHON_BIN"
test -d "$DATA_ROOT/train"
test -d "$DATA_ROOT/val"
test -d "$CODE_DIR"

if pgrep -u "$USER" -f "[t]rain.py" >/dev/null 2>&1; then
  echo "ERROR: another SHARP training process is active."
  echo "Stop or finish that run before starting this four-GPU suite."
  pgrep -af "[t]rain.py" || true
  exit 1
fi

echo "SHARP_AV2_20EPOCH_10TEST_START=$(date --iso-8601=seconds)"
echo "Experiment: $EXPERIMENT_ROOT"
echo "Results:    $RESULTS_ROOT"
echo "Control: global batch 32, 13 warm-up epochs, LR 1e-4 to 1e-5, seed 2333"
echo "Execution: four GPUs, per-GPU batch 8, six workers per process"

cd "$CODE_DIR"

if [ ! -s "$RESULTS_ROOT/PREFLIGHT_OK" ]; then
  echo "Running all-variant preflight checks..."
  "$PYTHON_BIN" "$EXPERIMENT_ROOT/preflight_suite.py" \
    2>&1 | tee "$RESULTS_ROOT/preflight.log"
  PREFLIGHT_STATUS=${PIPESTATUS[0]}
  if [ "$PREFLIGHT_STATUS" -ne 0 ]; then
    echo "ERROR: preflight failed. No training variant was started."
    exit "$PREFLIGHT_STATUS"
  fi
  printf 'completed=%s\n' "$(date --iso-8601=seconds)" > "$RESULTS_ROOT/PREFLIGHT_OK"
fi

VARIANTS=(
  "01_baseline:baseline"
  "02_confidence_gated_memory:confidence_gated_memory"
  "03_cross_window_consistency:cross_window_consistency"
  "04_learned_temporal_pool:learned_temporal_pool"
  "05_uncertainty_target_context:uncertainty_target_context"
  "06_relative_geometry_bias:relative_geometry_bias"
  "07_kinematic_motion_stem:kinematic_motion_stem"
  "08_endpoint_refinement_decoder:endpoint_refinement_decoder"
  "09_lane_topology_graph:lane_topology_graph"
  "10_agent_temporal_mamba:agent_temporal_mamba"
)

for ITEM in "${VARIANTS[@]}"; do
  DIRECTORY_NAME=${ITEM%%:*}
  VARIANT=${ITEM#*:}
  VARIANT_ROOT="$RESULTS_ROOT/$DIRECTORY_NAME"
  OUTPUT_DIR="$VARIANT_ROOT/run"
  LOG="$VARIANT_ROOT/full_run.log"
  mkdir -p "$VARIANT_ROOT" "$OUTPUT_DIR"

  if [ -s "$VARIANT_ROOT/COMPLETED" ]; then
    echo "SKIP_COMPLETE=$DIRECTORY_NAME"
    continue
  fi

  export SHARP_EXPERIMENT_VARIANT="$VARIANT"
  printf 'attempt_start=%s\n' "$(date --iso-8601=seconds)" >> "$VARIANT_ROOT/ATTEMPTS.log"
  printf 'variant=%s\n' "$VARIANT" > "$VARIANT_ROOT/VARIANT.txt"

  LAST_CHECKPOINT="$OUTPUT_DIR/checkpoints/last.ckpt"
  if [ -s "$LAST_CHECKPOINT" ]; then
    CHECKPOINT_OVERRIDE="checkpoint=$LAST_CHECKPOINT"
    echo "RESUMING_VARIANT=$VARIANT"
    echo "RESUME_CHECKPOINT=$LAST_CHECKPOINT"
  else
    CHECKPOINT_OVERRIDE="checkpoint=null"
    echo "STARTING_VARIANT=$VARIANT"
  fi

  COMMAND=(
    "$PYTHON_BIN" train.py
    "seed=2333"
    "gpus=4"
    "epochs=20"
    "batch_size=8"
    "output_dir=$OUTPUT_DIR"
    "datamodule.pl_module.data_root=$DATA_ROOT"
    "datamodule.pl_module.num_workers=6"
    "model.pl_module.optim.lr=0.0001"
    "model.pl_module.optim.min_lr=0.00001"
    "model.pl_module.optim.weight_decay=0.01"
    "model.pl_module.optim.warmup_ratio=0.65"
    "trainer.devices=4"
    "trainer.strategy=ddp_find_unused_parameters_false"
    "trainer.gradient_clip_val=5"
    "trainer.gradient_clip_algorithm=norm"
    "trainer.sync_batchnorm=true"
    "trainer.num_sanity_val_steps=0"
    "callbacks.0.save_top_k=-1"
    "callbacks.0.save_last=true"
    "callbacks.0.every_n_epochs=1"
    "$CHECKPOINT_OVERRIDE"
  )

  set +e
  "${COMMAND[@]}" 2>&1 | tee -a "$LOG"
  TRAIN_STATUS=${PIPESTATUS[0]}
  set -e

  printf 'attempt_end=%s status=%s\n' \
    "$(date --iso-8601=seconds)" "$TRAIN_STATUS" >> "$VARIANT_ROOT/ATTEMPTS.log"

  if [ "$TRAIN_STATUS" -ne 0 ]; then
    printf 'variant=%s\nstatus=%s\ntime=%s\n' \
      "$VARIANT" "$TRAIN_STATUS" "$(date --iso-8601=seconds)" \
      > "$VARIANT_ROOT/INTERRUPTED_OR_FAILED"
    echo "VARIANT_STOPPED=$VARIANT status=$TRAIN_STATUS"
    echo "Run this same suite script again; it will resume from last.ckpt."
    exit "$TRAIN_STATUS"
  fi

  "$PYTHON_BIN" "$EXPERIMENT_ROOT/summarize_variant.py" \
    "$VARIANT_ROOT" --variant "$VARIANT" | tee "$VARIANT_ROOT/finalize.log"
  printf 'variant=%s\ncompleted=%s\n' \
    "$VARIANT" "$(date --iso-8601=seconds)" > "$VARIANT_ROOT/COMPLETED"
  rm -f "$VARIANT_ROOT/INTERRUPTED_OR_FAILED"
  echo "VARIANT_COMPLETE=$VARIANT"
done

"$PYTHON_BIN" "$EXPERIMENT_ROOT/suite_status.py" "$RESULTS_ROOT" \
  | tee "$RESULTS_ROOT/FINAL_SUITE_STATUS.txt"
printf 'completed=%s\n' "$(date --iso-8601=seconds)" > "$RESULTS_ROOT/SUITE_COMPLETED"
echo "SHARP_AV2_20EPOCH_10TEST_COMPLETE=True"
