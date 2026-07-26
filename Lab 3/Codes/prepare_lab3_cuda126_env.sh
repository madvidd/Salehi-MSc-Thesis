#!/usr/bin/env bash
set -Eeuo pipefail

BASE=/home/server01/M
SOURCE_ENV="$BASE/Codes/AV2/envs/sharp_av2"
TARGET_ENV="$BASE/Codes/AV2/envs/sharp_av2_cu126"
CONDA="$BASE/Codes/AV2/miniforge3/bin/conda"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_POINTER="$BASE/Codes/LATEST_SHARP_ATTENTION_ENV.txt"

for required in "$CONDA" "$SOURCE_ENV/bin/python"; do
  if [[ ! -e "$required" ]]; then
    echo "FATAL: required path is missing: $required" >&2
    exit 1
  fi
done

if [[ ! -x "$TARGET_ENV/bin/python" ]]; then
  echo "Cloning the existing Lab 3 environment without sudo..."
  "$CONDA" create -y -p "$TARGET_ENV" --clone "$SOURCE_ENV"
fi

RUNTIME_OK=$("$TARGET_ENV/bin/python" - <<'PY' 2>/dev/null || true
import torch
ok = torch.__version__.startswith("2.8.0") and torch.version.cuda == "12.6"
print("yes" if ok else "no")
PY
)

if [[ "$RUNTIME_OK" != yes ]]; then
  echo "Installing the official PyTorch 2.8 CUDA 12.6 wheels..."
  "$TARGET_ENV/bin/python" -m pip install \
    --no-cache-dir \
    --force-reinstall \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126
fi

"$TARGET_ENV/bin/python" -m pip check
"$TARGET_ENV/bin/python" - <<'PY'
import subprocess

import torch

if not torch.__version__.startswith("2.8.0"):
    raise SystemExit(f"FATAL: expected PyTorch 2.8.0, found {torch.__version__}")
if torch.version.cuda != "12.6":
    raise SystemExit(f"FATAL: expected CUDA 12.6, found {torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
    raise SystemExit(
        f"FATAL: expected at least three CUDA GPUs, found {torch.cuda.device_count()}"
    )

driver = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
        "-i",
        "0",
    ],
    text=True,
).strip()
parts = tuple(int(part) for part in driver.split(".")[:3])
if parts < (560, 35, 3):
    raise SystemExit(
        f"FATAL: NVIDIA driver {driver} is older than required 560.35.03"
    )

print(f"PYTORCH_VERSION={torch.__version__}")
print(f"PYTORCH_CUDA_RUNTIME={torch.version.cuda}")
print(f"NVIDIA_DRIVER={driver}")
print(f"CUDA_DEVICE_COUNT={torch.cuda.device_count()}")
for index in range(3):
    print(f"GPU_{index}={torch.cuda.get_device_name(index)}")
PY

printf '%s\n' "$TARGET_ENV" > "$ENV_POINTER"
"$TARGET_ENV/bin/python" "$SCRIPT_DIR/prepare_remaining_attention_runtime.py"

export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=1

"$TARGET_ENV/bin/torchrun" \
  --standalone \
  --nproc_per_node=3 \
  "$SCRIPT_DIR/validate_lab3_cuda_runtime.py"

echo "LAB3_CUDA126_ENV_READY=$TARGET_ENV"
