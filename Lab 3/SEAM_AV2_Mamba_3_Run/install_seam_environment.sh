#!/usr/bin/env bash

set +e
set +u
set +o pipefail 2>/dev/null

BASE=${BASE:-/home/server01/M}
ENV_DIR="$BASE/Codes/envs/seam_av2_mamba_torch211"
PACKAGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
READY="$ENV_DIR/.seam_environment_ready_v1"
STATUS=0

if [ ! -x "$ENV_DIR/bin/python" ]; then
  CONDA=$(command -v conda)
  MAMBA=$(command -v mamba)
  MICROMAMBA=$(command -v micromamba)

  if [ -n "$MAMBA" ]; then
    "$MAMBA" create -y -p "$ENV_DIR" python=3.11.10 pip
  elif [ -n "$CONDA" ]; then
    "$CONDA" create -y -p "$ENV_DIR" python=3.11.10 pip
  elif [ -n "$MICROMAMBA" ]; then
    "$MICROMAMBA" create -y -p "$ENV_DIR" python=3.11.10 pip
  else
    echo "ERROR: conda, mamba, or micromamba is required to create Python 3.11.10."
    STATUS=1
  fi
fi

if [ "$STATUS" -eq 0 ] && [ ! -f "$READY" ]; then
  PYTHON="$ENV_DIR/bin/python"
  export MAX_JOBS=${MAX_JOBS:-$(nproc)}
  export TORCH_CUDA_ARCH_LIST=7.5

  "$PYTHON" -m pip install --upgrade \
    "pip==24.2" "setuptools==69.5.1" "wheel==0.44.0" \
    "packaging==24.1" "ninja==1.11.1.1"
  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    "$PYTHON" -m pip install \
      torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
      --index-url https://download.pytorch.org/whl/cu121
    STATUS=$?
  fi

  if [ "$STATUS" -eq 0 ]; then
    "$PYTHON" -m pip install \
      -r "$PACKAGE_DIR/requirements-lab3.txt" \
      --no-build-isolation
    STATUS=$?
  fi

  if [ "$STATUS" -eq 0 ]; then
    "$PYTHON" - <<'PY'
import torch
import pytorch_lightning
import mamba_ssm

assert torch.__version__.startswith("2.1.1"), torch.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert torch.cuda.is_available()
print("PYTORCH_VERSION=" + torch.__version__)
print("PYTORCH_CUDA=" + str(torch.version.cuda))
print("LIGHTNING_VERSION=" + pytorch_lightning.__version__)
print("MAMBA_IMPORT_OK=True")
PY
    STATUS=$?
  fi

  if [ "$STATUS" -eq 0 ]; then
    date --iso-8601=seconds > "$READY"
  fi
fi

if [ "$STATUS" -eq 0 ]; then
  echo "SEAM_ENVIRONMENT_READY=$ENV_DIR"
else
  echo "ERROR: SEAM environment installation did not complete."
fi

echo "Environment status: $STATUS"
echo "Terminal remains open."
exit "$STATUS"
