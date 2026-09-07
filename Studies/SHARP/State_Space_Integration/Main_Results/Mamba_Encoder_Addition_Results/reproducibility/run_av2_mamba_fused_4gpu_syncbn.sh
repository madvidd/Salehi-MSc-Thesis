#!/usr/bin/env bash
set -euo pipefail

LAB_BASE="/home/server00/M"
EXPERIMENT_ROOT="/home/server00/M/Codes/SHARP_AV2_MAMBA_FUSED80_20260717-120414"
CODE_DIR="$EXPERIMENT_ROOT/Code"
RESULTS_ROOT="/home/server00/M/Results/SHARP_AV2_MAMBA_FUSED80_20260717-120414"
ATTEMPT_ID=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="$RESULTS_ROOT/$ATTEMPT_ID"
DATA_DIR="$LAB_BASE/Datasets/AV2/sharp_processed"
PYTHON_BIN="$LAB_BASE/Codes/envs/sharp/bin/python"
FUSED_PACKAGES="$EXPERIMENT_ROOT/fused_packages"
BATCH_SIZE="${BATCH_SIZE:-8}"
CPU_COUNT=$(nproc)
AUTO_WORKERS=$(( (CPU_COUNT - 4) / 4 ))
if [ "$AUTO_WORKERS" -lt 4 ]; then AUTO_WORKERS=4; fi
if [ "$AUTO_WORKERS" -gt 12 ]; then AUTO_WORKERS=12; fi
WORKERS="${WORKERS:-$AUTO_WORKERS}"

mkdir -p "$RESULTS_DIR"
printf '%s\n' "$RESULTS_DIR" > "$LAB_BASE/Results/LATEST_SHARP_AV2_MAMBA_FUSED80_RUN.txt"

source "$LAB_BASE/Codes/miniforge3/etc/profile.d/conda.sh"
conda activate "$LAB_BASE/Codes/envs/sharp"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$FUSED_PACKAGES:$CODE_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ulimit -n 65535 2>/dev/null || true

test -d "$DATA_DIR/train"
test -d "$DATA_DIR/val"
test -f "$EXPERIMENT_ROOT/FUSED_INSTALL_OK"
test -f "$FUSED_PACKAGES/selective_scan_cuda.cpython-311-x86_64-linux-gnu.so"
test -f "$FUSED_PACKAGES/causal_conv1d_cuda.cpython-311-x86_64-linux-gnu.so"

if pgrep -u "$USER" -f "$PYTHON_BIN.*train.py" >/dev/null 2>&1; then
  echo "ERROR: another SHARP training process is already running."
  echo "Wait for it to finish or stop it explicitly before using all four GPUs."
  ps -u "$USER" -o pid,etime,%cpu,%mem,cmd | grep -E "train.py" | grep -v grep || true
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected 4 CUDA GPUs, found {torch.cuda.device_count()}")
print("CUDA GPUs:")
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

cd "$CODE_DIR"

# Stress the stable split fused CUDA path before the 80-epoch run.
"$PYTHON_BIN" - <<'PY'
import time
import torch
import causal_conv1d_cuda
import selective_scan_cuda
import mamba_ssm.modules.mamba_simple as mamba_simple
from src.model.layers.mamba_encoder import SceneMambaEncoder, fused_cuda_available

if not fused_cuda_available():
    raise SystemExit("Fused CUDA Mamba is unavailable")

calls = {"conv": 0, "scan": 0}
original_conv = mamba_simple.causal_conv1d_fn
original_scan = mamba_simple.selective_scan_fn

def tracked_conv(*args, **kwargs):
    calls["conv"] += 1
    return original_conv(*args, **kwargs)

def tracked_scan(*args, **kwargs):
    calls["scan"] += 1
    return original_scan(*args, **kwargs)

mamba_simple.causal_conv1d_fn = tracked_conv
mamba_simple.selective_scan_fn = tracked_scan
device = torch.device("cuda:0")
module = SceneMambaEncoder(dim=128, depth=1).to(device)
lengths = (64, 96, 128, 160, 192, 224, 256, 320)
stress_iterations = 160
last_shape = None
last_grad_count = 0
for iteration in range(stress_iterations):
    length = lengths[iteration % len(lengths)]
    x = torch.randn(8, length, 128, device=device, requires_grad=True)
    mask = torch.ones(8, length, dtype=torch.bool, device=device)
    mask[:, -(iteration % 17 + 1):] = False
    y = module(x, mask)
    if not torch.isfinite(y).all():
        raise SystemExit(f"Non-finite split fused output at iteration {iteration}")
    y.square().mean().backward()
    last_grad_count = sum(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in module.parameters()
    )
    if last_grad_count == 0:
        raise SystemExit(f"Missing gradients at stress iteration {iteration}")
    module.zero_grad(set_to_none=True)
    last_shape = tuple(y.shape)
torch.cuda.synchronize()
expected_calls = stress_iterations * 2
if calls["conv"] != expected_calls or calls["scan"] != expected_calls:
    raise SystemExit(
        f"Split fused kernels were not called as expected: {calls}, "
        f"expected={expected_calls}"
    )

module.eval()
sample = torch.randn(8, 128, 128, device=device)
mask = torch.ones(8, 128, dtype=torch.bool, device=device)
mask[:, -8:] = False
with torch.no_grad():
    for _ in range(5):
        module(sample, mask)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(25):
        module(sample, mask)
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start
param_count = sum(parameter.numel() for parameter in module.parameters())
print("causal_conv1d_cuda:", causal_conv1d_cuda.__file__)
print("selective_scan_cuda:", selective_scan_cuda.__file__)
print(
    f"FUSED_MAMBA_CUDA_SMOKE_TEST_OK shape={last_shape} "
    f"params={param_count} grads={last_grad_count} "
    f"conv_calls={calls['conv']} scan_calls={calls['scan']} "
    f"forward_ms={elapsed * 1000 / 25:.3f} split_kernels=True"
)
print(
    f"FUSED_SPLIT_MAMBA_CUDA_STRESS_OK iterations={stress_iterations} "
    f"lengths={lengths}"
)
PY
"$PYTHON_BIN" - <<'PY' > "$RESULTS_DIR/environment.txt"
import platform
import torch
import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("mamba_ssm:", mamba_ssm.__version__)
print("causal_conv1d:", causal_conv1d.__version__)
print("causal_conv1d_cuda:", causal_conv1d_cuda.__file__)
print("selective_scan_cuda:", selective_scan_cuda.__file__)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

echo "Starting SHARP + stable split-kernel fused CUDA Mamba on four GPUs"
echo "Per-GPU batch size: $BATCH_SIZE; global batch size: $((BATCH_SIZE * 4))"
echo "CPU threads: $CPU_COUNT; DataLoader workers per process: $WORKERS; total workers: $((WORKERS * 4))"
echo "Article-aligned schedule: 80 epochs, 13 warm-up epochs, LR 1e-4 -> 1e-5"

"$PYTHON_BIN" -u train.py \
  seed=2333 \
  gpus=4 \
  epochs=80 \
  batch_size="$BATCH_SIZE" \
  output_dir="$RESULTS_DIR/run" \
  datamodule.pl_module.data_root="$DATA_DIR" \
  datamodule.pl_module.num_workers="$WORKERS" \
  trainer.devices=4 \
  trainer.strategy=ddp_find_unused_parameters_false \
  trainer.sync_batchnorm=true \
  trainer.num_sanity_val_steps=0 \
  callbacks.0.save_top_k=3 \
  +callbacks.0.save_last=true \
  callbacks.0.monitor=minADE6 \
  2>&1 | tee "$RESULTS_DIR/full_run.log"

BEST_CHECKPOINT=$("$PYTHON_BIN" - "$RESULTS_DIR/run/checkpoints" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
candidates = []
for checkpoint in root.glob("*.ckpt"):
    match = re.search(r"minADE6_([0-9.]+)\.ckpt$", checkpoint.name)
    if match:
        candidates.append((float(match.group(1)), checkpoint))
if not candidates:
    raise SystemExit("No minADE6 checkpoint found")
print(min(candidates, key=lambda item: item[0])[1])
PY
)

echo "Best checkpoint: $BEST_CHECKPOINT"
"$PYTHON_BIN" "$EXPERIMENT_ROOT/verify_mamba_checkpoint.py" "$BEST_CHECKPOINT" \
  | tee "$RESULTS_DIR/mamba_checkpoint_verification.txt"

echo "SHARP_MAMBA_RUN_COMPLETE"
echo "RESULTS_DIR=$RESULTS_DIR"
