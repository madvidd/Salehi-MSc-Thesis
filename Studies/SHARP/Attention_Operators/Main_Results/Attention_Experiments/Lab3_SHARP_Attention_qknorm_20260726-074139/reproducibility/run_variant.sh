#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?usage: run_variant.sh VARIANT}"
case "$VARIANT" in
  baseline_mha|qknorm|talking_heads|qknorm_talking_heads) ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

BASE=/home/server01/M
ROOT=/home/server01/M/Codes/SHARP_ATTENTION_ABLATION_20260717-123916
RESULTS_ROOT=/home/server01/M/Results/SHARP_ATTENTION_ABLATION_20260717-123916
CODE="$ROOT/variants/$VARIANT/Code"
OUT="$RESULTS_ROOT/$VARIANT"
ENV="$BASE/Codes/AV2/envs/sharp_av2"
DATA="$BASE/Datasets/AV2/sharp_processed"
PATCH=/home/server01/M/Codes/SHARP_ATTENTION_ABLATION_20260717-123916/runtime_patch

mkdir -p "$OUT/run" "$OUT/eval"
source "$BASE/Codes/AV2/miniforge3/etc/profile.d/conda.sh"
conda activate "$ENV"

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH="$PATCH:$CODE:${PYTHONPATH:-}"
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export NO_COLOR=1
export RICH_NO_COLOR=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CPU_COUNT=$(nproc)
DEFAULT_WORKERS=$(( (CPU_COUNT - 3) / 3 ))
(( DEFAULT_WORKERS < 4 )) && DEFAULT_WORKERS=4
(( DEFAULT_WORKERS > 8 )) && DEFAULT_WORKERS=8
WORKERS_PER_RANK=${WORKERS_PER_RANK:-$DEFAULT_WORKERS}
BATCH_PER_GPU=${BATCH_PER_GPU:-8}
EPOCHS=${EPOCHS:-80}
GLOBAL_BATCH=$(( BATCH_PER_GPU * 3 ))

echo "VARIANT=$VARIANT"
echo "CODE=$CODE"
echo "RESULTS=$OUT"
echo "GPUS=3 BATCH_PER_GPU=$BATCH_PER_GPU GLOBAL_BATCH=$GLOBAL_BATCH"
echo "WORKERS_PER_RANK=$WORKERS_PER_RANK TOTAL_WORKERS=$((WORKERS_PER_RANK * 3))"
nvidia-smi

if [[ -f "$OUT/COMPLETE" && -f "$OUT/metrics.json" ]]; then
  echo "$VARIANT is already complete; skipping training."
  exit 0
fi

cd "$CODE"
ARGS=(
  seed=2333
  gpus=3
  epochs="$EPOCHS"
  batch_size="$BATCH_PER_GPU"
  output_dir="$OUT/run"
  datamodule.pl_module.data_root="$DATA"
  datamodule.pl_module.num_workers="$WORKERS_PER_RANK"
  model.pl_module.optim.lr=0.0001
  model.pl_module.optim.min_lr=0.00001
  model.pl_module.optim.warmup_ratio=0.1625
  trainer.devices=3
  trainer.strategy=ddp_find_unused_parameters_false
  trainer.sync_batchnorm=true
  +trainer.num_sanity_val_steps=0
  callbacks.0.save_top_k=3
  callbacks.0.monitor=minADE6
)

LAST="$OUT/run/checkpoints/last.ckpt"
if [[ -f "$LAST" ]]; then
  echo "Resuming $VARIANT from $LAST"
  ARGS+=(checkpoint="$LAST")
fi

set +e
python -u train.py "${ARGS[@]}" 2>&1 | tee -a "$OUT/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
if (( TRAIN_RC != 0 )); then
  echo "Training failed for $VARIANT with exit code $TRAIN_RC" >&2
  exit "$TRAIN_RC"
fi

BEST=$(python "$ROOT/find_best_checkpoint.py" "$OUT/run/checkpoints")
echo "$BEST" > "$OUT/best_checkpoint.txt"
export SHARP_METRICS_JSON="$OUT/metrics.json"

python -u eval_to_json.py   seed=2333   gpus=3   batch_size="$BATCH_PER_GPU"   output_dir="$OUT/eval"   checkpoint="$BEST"   datamodule.pl_module.data_root="$DATA"   datamodule.pl_module.num_workers="$WORKERS_PER_RANK"   2>&1 | tee "$OUT/eval.log"

date --iso-8601=seconds > "$OUT/COMPLETE"
echo "Completed $VARIANT"
