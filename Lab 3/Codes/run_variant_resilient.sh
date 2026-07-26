#!/usr/bin/env bash
set -Eeuo pipefail

VARIANT="${1:?usage: run_variant.sh VARIANT}"
case "$VARIANT" in
  baseline_mha|qknorm|talking_heads|qknorm_talking_heads) ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

BASE=/home/server01/M
ROOT=$(tr -d '\r\n' < "$BASE/Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")
RESULTS_ROOT=$(tr -d '\r\n' < "$BASE/Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
ENV_POINTER="$BASE/Codes/LATEST_SHARP_ATTENTION_ENV.txt"
ENV=$(tr -d '\r\n' < "$ENV_POINTER")
CODE="$ROOT/variants/$VARIANT/Code"
OUT="$RESULTS_ROOT/$VARIANT"
DATA="$BASE/Datasets/AV2/sharp_processed"
PATCH="$ROOT/runtime_patch"

for required in "$ENV/bin/python" "$CODE/train.py" "$DATA" "$PATCH/sitecustomize.py"; do
  if [[ ! -e "$required" ]]; then
    echo "FATAL: required runtime path is missing: $required" >&2
    exit 1
  fi
done

mkdir -p "$OUT/run" "$OUT/eval" "$OUT/diagnostics"

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH="$PATCH:$CODE:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
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
export PYTORCH_ALLOC_CONF=max_split_size_mb:128
export CUDA_MODULE_LOADING=LAZY
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
echo "ENV=$ENV"
echo "GPUS=3 BATCH_PER_GPU=$BATCH_PER_GPU GLOBAL_BATCH=$GLOBAL_BATCH"
echo "WORKERS_PER_RANK=$WORKERS_PER_RANK TOTAL_WORKERS=$((WORKERS_PER_RANK * 3))"
"$ENV/bin/python" - <<'PY'
import torch

print(f"PYTORCH_VERSION={torch.__version__}")
print(f"PYTORCH_CUDA_RUNTIME={torch.version.cuda}")
print(f"CUDA_DEVICE_COUNT={torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
    raise SystemExit("FATAL: three CUDA devices are required")
if not torch.__version__.startswith("2.8.0") or torch.version.cuda != "12.6":
    raise SystemExit("FATAL: expected PyTorch 2.8.0 with CUDA 12.6")
PY
nvidia-smi

if [[ -f "$OUT/COMPLETE" && -s "$OUT/metrics.json" ]]; then
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

START_EPOCH=$(date +%s)
printf '%s\n' "$START_EPOCH" > "$OUT/diagnostics/training_start_epoch.txt"
set +e
"$ENV/bin/python" -u train.py "${ARGS[@]}" 2>&1 | tee -a "$OUT/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e

if (( TRAIN_RC != 0 )); then
  {
    echo "variant=$VARIANT"
    echo "exit_code=$TRAIN_RC"
    echo "failed=$(date --iso-8601=seconds)"
    echo "pytorch=$("$ENV/bin/python" -c 'import torch; print(torch.__version__)')"
    echo "cuda_runtime=$("$ENV/bin/python" -c 'import torch; print(torch.version.cuda)')"
    echo
    echo "===== KERNEL EVENTS SINCE TRAINING START ====="
    journalctl -k --since "@$START_EPOCH" --no-pager 2>&1 \
      | grep -Ei 'segfault|libc10_cuda|NVRM: Xid|oom-kill|out of memory' \
      || echo "No matching kernel event was readable."
    echo
    echo "===== GPU STATE ====="
    nvidia-smi 2>&1
  } > "$OUT/diagnostics/failure_$(date +%Y%m%d-%H%M%S).txt"
  echo "Training failed for $VARIANT with exit code $TRAIN_RC" >&2
  exit "$TRAIN_RC"
fi

BEST=$("$ENV/bin/python" "$ROOT/find_best_checkpoint.py" "$OUT/run/checkpoints")
if [[ ! -f "$BEST" ]]; then
  echo "FATAL: best checkpoint was not found: $BEST" >&2
  exit 1
fi
echo "$BEST" > "$OUT/best_checkpoint.txt"
export SHARP_METRICS_JSON="$OUT/metrics.json"

set +e
"$ENV/bin/python" -u eval_to_json.py \
  seed=2333 \
  gpus=3 \
  batch_size="$BATCH_PER_GPU" \
  output_dir="$OUT/eval" \
  checkpoint="$BEST" \
  datamodule.pl_module.data_root="$DATA" \
  datamodule.pl_module.num_workers="$WORKERS_PER_RANK" \
  2>&1 | tee "$OUT/eval.log"
EVAL_RC=${PIPESTATUS[0]}
set -e
if (( EVAL_RC != 0 )); then
  echo "Evaluation failed for $VARIANT with exit code $EVAL_RC" >&2
  exit "$EVAL_RC"
fi

"$ENV/bin/python" - "$OUT/metrics.json" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(f"FATAL: missing metrics file: {path}")
data = json.loads(path.read_text())
if not data:
    raise SystemExit(f"FATAL: metrics file is empty: {path}")
for key, value in data.items():
    if isinstance(value, (int, float)) and not math.isfinite(value):
        raise SystemExit(f"FATAL: non-finite metric {key}={value}")
print(f"METRICS_VALID={path}")
PY

date --iso-8601=seconds > "$OUT/COMPLETE"
echo "Completed $VARIANT"
