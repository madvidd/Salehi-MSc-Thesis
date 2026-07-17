#!/usr/bin/env python3
"""Create an article-aligned 80-epoch SHARP + fused CUDA Mamba experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import stat
from pathlib import Path


MAMBA_ENCODER_SOURCE = r'''"""Official fused CUDA Mamba encoder for SHARP scene tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import causal_conv1d_cuda
import selective_scan_cuda
import mamba_ssm.modules.mamba_simple as mamba_simple
from mamba_ssm.modules.mamba_simple import Mamba


def fused_cuda_available() -> bool:
    return (
        mamba_simple.causal_conv1d_fn is not None
        and mamba_simple.mamba_inner_fn is not None
        and hasattr(causal_conv1d_cuda, "causal_conv1d_fwd")
        and hasattr(selective_scan_cuda, "fwd")
    )


class SceneMambaEncoder(nn.Module):
    """Bidirectional fused Mamba residual encoder for SHARP scene tokens."""

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if not fused_cuda_available():
            raise RuntimeError("Official fused Mamba CUDA kernels are unavailable")
        self.bidirectional = bidirectional
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.layers = nn.ModuleList(
            Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=True,
            )
            for _ in range(depth)
        )
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(depth))
        # sigmoid(-2.1972) = 0.1: begin close to baseline SHARP.
        self.residual_gates = nn.ParameterList(
            nn.Parameter(torch.tensor(-2.1972246)) for _ in range(depth)
        )

    def reset_mamba_parameters(self) -> None:
        """Restore official Mamba delta initialization after SHARP init."""
        for layer in self.layers:
            dt = torch.exp(
                torch.rand(layer.d_inner, device=layer.dt_proj.bias.device)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp_min(1e-4)
            inverse_softplus = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                layer.dt_proj.bias.copy_(inverse_softplus)
            layer.dt_proj.bias._no_reinit = True

    def forward(
        self,
        x: torch.Tensor,
        key_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = key_valid_mask.unsqueeze(-1).to(dtype=x.dtype)

        for norm, layer, dropout, residual_gate in zip(
            self.norms,
            self.layers,
            self.dropouts,
            self.residual_gates,
        ):
            h = norm(x) * valid
            forward_out = layer(h)
            if self.bidirectional:
                backward_out = torch.flip(layer(torch.flip(h, dims=[1])), dims=[1])
                update = 0.5 * (forward_out + backward_out)
            else:
                update = forward_out
            x = x + torch.sigmoid(residual_gate) * dropout(update) * valid

        return x
'''

VERIFY_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    keys = [key for key in state_dict if "mamba_encoder" in key]
    count = sum(state_dict[key].numel() for key in keys)

    print(f"MAMBA_CHECKPOINT_KEYS={len(keys)}")
    print(f"MAMBA_CHECKPOINT_PARAMETERS={count}")
    for key in keys[:20]:
        print(key)

    if not keys or count == 0:
        raise SystemExit("ERROR: checkpoint contains no Mamba parameters")


if __name__ == "__main__":
    main()
'''


INSTALL_SCRIPT_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail

LAB_BASE="{base}"
EXPERIMENT_ROOT="{experiment_root}"
PYTHON_BIN="$LAB_BASE/Codes/envs/sharp/bin/python"
WHEEL_DIR="$EXPERIMENT_ROOT/fused_wheels"
VENDOR_DIR="$EXPERIMENT_ROOT/fused_packages"

CAUSAL_NAME="causal_conv1d-1.6.2.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
MAMBA_NAME="mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
CAUSAL_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
MAMBA_URL="https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
CAUSAL_SHA256="ca0bca912bf9ad3761f3a900815d0940c1d457c1df27a1c024b5ae43a0cfd68b"
MAMBA_SHA256="413bda17da4222fda3513e56814a98a9026bcf1be95191baddb8735b3688a788"

source "$LAB_BASE/Codes/miniforge3/etc/profile.d/conda.sh"
conda activate "$LAB_BASE/Codes/envs/sharp"

if pgrep -u "$USER" -f "$PYTHON_BIN.*train.py" >/dev/null 2>&1; then
  echo "ERROR: stop the current SHARP run before installing fused Mamba."
  ps -u "$USER" -o pid,etime,%cpu,%mem,cmd | grep -E "train.py" | grep -v grep || true
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch_before:", torch.__version__)
print("torch_cuda_before:", torch.version.cuda)
print("cxx11_abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
if sys.version_info[:2] != (3, 11):
    raise SystemExit("Expected Python 3.11")
if not torch.__version__.startswith("2.8.") or not torch.version.cuda.startswith("12.8"):
    raise SystemExit("Expected Torch 2.8 with CUDA 12.8")
if not torch._C._GLIBCXX_USE_CXX11_ABI:
    raise SystemExit("Official fused wheels require CXX11 ABI TRUE")
PY

mkdir -p "$WHEEL_DIR" "$VENDOR_DIR"

download_verified() {{
  local url="$1"
  local output="$2"
  local expected="$3"
  if [ -f "$output" ] && printf '%s  %s\n' "$expected" "$output" | sha256sum -c - >/dev/null 2>&1; then
    echo "Verified cached wheel: $output"
    return
  fi
  rm -f "$output.part"
  curl --fail --location --retry 5 --retry-delay 3 --output "$output.part" "$url"
  printf '%s  %s\n' "$expected" "$output.part" | sha256sum -c -
  mv "$output.part" "$output"
}}

download_verified "$CAUSAL_URL" "$WHEEL_DIR/$CAUSAL_NAME" "$CAUSAL_SHA256"
download_verified "$MAMBA_URL" "$WHEEL_DIR/$MAMBA_NAME" "$MAMBA_SHA256"

"$PYTHON_BIN" -m pip install \
  --no-deps \
  --upgrade \
  --target "$VENDOR_DIR" \
  "einops==0.8.1" \
  "$WHEEL_DIR/$CAUSAL_NAME" \
  "$WHEEL_DIR/$MAMBA_NAME"

# Limit package initialization to Mamba-1 so optional Mamba-3 dependencies are
# not imported into the stable SHARP Torch 2.8 environment.
"$PYTHON_BIN" - "$VENDOR_DIR/mamba_ssm/__init__.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.write_text(
    '__version__ = "2.3.2.post1"\n'
    'from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn\n'
    'from mamba_ssm.modules.mamba_simple import Mamba\n',
    encoding="utf-8",
)
PY

PYTHONPATH="$VENDOR_DIR:${{PYTHONPATH:-}}" "$PYTHON_BIN" - <<'PY'
import torch
import causal_conv1d_cuda
import selective_scan_cuda
import mamba_ssm.modules.mamba_simple as mamba_simple
from mamba_ssm import Mamba

print("torch_after:", torch.__version__)
print("torch_cuda_after:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0))
print("compute_capability:", torch.cuda.get_device_capability(0))
print("causal_conv1d_cuda:", causal_conv1d_cuda.__file__)
print("selective_scan_cuda:", selective_scan_cuda.__file__)

if torch.__version__ != "2.8.0+cu128":
    raise SystemExit("Torch changed unexpectedly")
if torch.cuda.get_device_capability(0) != (7, 5):
    raise SystemExit("Expected RTX 2080 Ti compute capability 7.5")
if mamba_simple.causal_conv1d_fn is None or mamba_simple.mamba_inner_fn is None:
    raise SystemExit("Official Mamba fast path is unavailable")

calls = {{"count": 0}}
original = mamba_simple.mamba_inner_fn

def tracked(*args, **kwargs):
    calls["count"] += 1
    return original(*args, **kwargs)

mamba_simple.mamba_inner_fn = tracked
model = Mamba(d_model=128, d_state=16, d_conv=4, expand=2, use_fast_path=True).cuda()
x = torch.randn(4, 96, 128, device="cuda", requires_grad=True)
y = model(x)
y.square().mean().backward()
torch.cuda.synchronize()
if calls["count"] != 1:
    raise SystemExit(f"Fused Mamba path was not called exactly once: {{calls['count']}}")
if not torch.isfinite(y).all():
    raise SystemExit("Fused Mamba produced non-finite output")
print("FUSED_CUDA_MAMBA_INSTALL_OK", tuple(y.shape), "fast_path_calls=", calls["count"])
PY

touch "$EXPERIMENT_ROOT/FUSED_INSTALL_OK"
echo "FUSED_INSTALL_COMPLETE"
echo "VENDOR_DIR=$VENDOR_DIR"
'''

RUN_SCRIPT_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail

LAB_BASE="{base}"
EXPERIMENT_ROOT="{experiment_root}"
CODE_DIR="$EXPERIMENT_ROOT/Code"
RESULTS_ROOT="{results_root}"
ATTEMPT_ID=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="$RESULTS_ROOT/$ATTEMPT_ID"
DATA_DIR="$LAB_BASE/Datasets/AV2/sharp_processed"
PYTHON_BIN="$LAB_BASE/Codes/envs/sharp/bin/python"
FUSED_PACKAGES="$EXPERIMENT_ROOT/fused_packages"
BATCH_SIZE="${{BATCH_SIZE:-8}}"
CPU_COUNT=$(nproc)
AUTO_WORKERS=$(( (CPU_COUNT - 4) / 4 ))
if [ "$AUTO_WORKERS" -lt 4 ]; then AUTO_WORKERS=4; fi
if [ "$AUTO_WORKERS" -gt 12 ]; then AUTO_WORKERS=12; fi
WORKERS="${{WORKERS:-$AUTO_WORKERS}}"

mkdir -p "$RESULTS_DIR"
printf '%s\n' "$RESULTS_DIR" > "$LAB_BASE/Results/LATEST_SHARP_AV2_MAMBA_FUSED80_RUN.txt"

source "$LAB_BASE/Codes/miniforge3/etc/profile.d/conda.sh"
conda activate "$LAB_BASE/Codes/envs/sharp"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$FUSED_PACKAGES:$CODE_DIR:${{PYTHONPATH:-}}"
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
    raise SystemExit(f"Expected 4 CUDA GPUs, found {{torch.cuda.device_count()}}")
print("CUDA GPUs:")
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

cd "$CODE_DIR"

# Prove that the official fused CUDA path executes before the 80-epoch run.
"$PYTHON_BIN" - <<'PY'
import time
import torch
import causal_conv1d_cuda
import selective_scan_cuda
import mamba_ssm.modules.mamba_simple as mamba_simple
from src.model.layers.mamba_encoder import SceneMambaEncoder, fused_cuda_available

if not fused_cuda_available():
    raise SystemExit("Fused CUDA Mamba is unavailable")

calls = {{"count": 0}}
original = mamba_simple.mamba_inner_fn

def tracked(*args, **kwargs):
    calls["count"] += 1
    return original(*args, **kwargs)

mamba_simple.mamba_inner_fn = tracked
device = torch.device("cuda:0")
module = SceneMambaEncoder(dim=128, depth=1).to(device)
x = torch.randn(4, 96, 128, device=device, requires_grad=True)
mask = torch.ones(4, 96, dtype=torch.bool, device=device)
mask[:, -8:] = False
y = module(x, mask)
y.square().mean().backward()
torch.cuda.synchronize()
grad_count = sum(
    parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
    for parameter in module.parameters()
)
if y.shape != x.shape or grad_count == 0 or calls["count"] < 2:
    raise SystemExit("Fused bidirectional Mamba smoke test failed")

module.eval()
with torch.no_grad():
    sample = x.detach()
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
    f"FUSED_MAMBA_CUDA_SMOKE_TEST_OK shape={{tuple(y.shape)}} "
    f"params={{param_count}} grads={{grad_count}} fast_path_calls={{calls['count']}} "
    f"forward_ms={{elapsed * 1000 / 25:.3f}}"
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

echo "Starting SHARP + official fused CUDA Mamba on four GPUs"
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
  trainer.sync_batchnorm=false \
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
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace("\r\n", "\n"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/server00/M")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    source_code = base / "Codes/SHARP/Code"
    dataset = base / "Datasets/AV2/sharp_processed"
    python_bin = base / "Codes/envs/sharp/bin/python"
    if not source_code.is_dir():
        raise SystemExit(f"Missing source SHARP code: {source_code}")
    if not dataset.joinpath("train").is_dir() or not dataset.joinpath("val").is_dir():
        raise SystemExit(f"Missing processed AV2 train/val data: {dataset}")
    if not python_bin.is_file():
        raise SystemExit(f"Missing SHARP Python: {python_bin}")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_root = base / "Codes" / f"SHARP_AV2_MAMBA_FUSED80_{stamp}"
    code_dir = experiment_root / "Code"
    results_root = base / "Results" / f"SHARP_AV2_MAMBA_FUSED80_{stamp}"
    experiment_root.mkdir(parents=True, exist_ok=False)
    results_root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        source_code,
        code_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".hydra", "outputs"),
    )

    layer_path = code_dir / "src/model/layers/mamba_encoder.py"
    layer_path.write_text(MAMBA_ENCODER_SOURCE, encoding="utf-8")

    sharp_path = code_dir / "src/model/sharp.py"
    sharp = sharp_path.read_text(encoding="utf-8")
    sharp = replace_once(
        sharp,
        "from .layers.multimodal_decoder_attn import MultimodalDecoder            \n",
        "from .layers.multimodal_decoder_attn import MultimodalDecoder            \n"
        "from .layers.mamba_encoder import SceneMambaEncoder\n",
        "Mamba import",
    )
    sharp = replace_once(
        sharp,
        "        dm=\"av2\",\n        k=6\n",
        "        dm=\"av2\",\n"
        "        k=6,\n"
        "        mamba_depth=1,\n"
        "        mamba_d_state=16,\n"
        "        mamba_d_conv=4,\n"
        "        mamba_expand=2,\n"
        "        mamba_dropout=0.05,\n"
        "        mamba_bidirectional=True\n",
        "Sharp constructor arguments",
    )
    sharp = replace_once(
        sharp,
        "        self.norm = nn.LayerNorm(embed_dim)\n\n",
        "        self.norm = nn.LayerNorm(embed_dim)\n\n"
        "        # Mamba scene encoder: after token construction and before streaming memory.\n"
        "        self.mamba_encoder = SceneMambaEncoder(\n"
        "            dim=embed_dim,\n"
        "            depth=mamba_depth,\n"
        "            d_state=mamba_d_state,\n"
        "            d_conv=mamba_d_conv,\n"
        "            expand=mamba_expand,\n"
        "            dropout=mamba_dropout,\n"
        "            bidirectional=mamba_bidirectional,\n"
        "        )\n\n",
        "Mamba module construction",
    )
    sharp = replace_once(
        sharp,
        "        x_encoder = x_encoder + pos_embed\n\n        #################\n        # DUAL TRAINING #\n",
        "        x_encoder = x_encoder + pos_embed\n"
        "        x_encoder = self.mamba_encoder(x_encoder, key_valid_mask)\n\n"
        "        #################\n        # DUAL TRAINING #\n",
        "Mamba forward insertion",
    )
    sharp = replace_once(
        sharp,
        "        self.initialize_weights()\n        return\n",
        "        self.initialize_weights()\n"
        "        # SHARP reinitializes Linear layers, so restore Mamba delta bias.\n"
        "        self.mamba_encoder.reset_mamba_parameters()\n"
        "        return\n",
        "Mamba post-SHARP initialization",
    )
    sharp_path.write_text(sharp, encoding="utf-8")

    model_cfg_path = code_dir / "conf/model/Sharp_av2.yaml"
    model_cfg = model_cfg_path.read_text(encoding="utf-8")
    model_cfg = replace_once(
        model_cfg,
        "    drop_path: 0.2\n",
        "    drop_path: 0.2\n"
        "    mamba_depth: 1\n"
        "    mamba_d_state: 16\n"
        "    mamba_d_conv: 4\n"
        "    mamba_expand: 2\n"
        "    mamba_dropout: 0.05\n"
        "    mamba_bidirectional: True\n",
        "Mamba model configuration",
    )
    model_cfg = replace_once(
        model_cfg,
        "    lr: 1e-3\n",
        "    lr: 1e-4\n",
        "article learning rate",
    )
    model_cfg_path.write_text(model_cfg, encoding="utf-8")

    config_path = code_dir / "conf/config.yaml"
    config = config_path.read_text(encoding="utf-8")
    config = config.replace(
        "  - _target_: pytorch_lightning.callbacks.RichProgressBar\n",
        "",
    )
    if "num_sanity_val_steps:" not in config:
        config = replace_once(
            config,
            "trainer:\n",
            "trainer:\n  num_sanity_val_steps: 0\n",
            "trainer configuration",
        )
    config_path.write_text(config, encoding="utf-8")

    datamodule_path = code_dir / "src/datamodules/av2_datamodule.py"
    datamodule = datamodule_path.read_text(encoding="utf-8")
    datamodule = datamodule.replace(
        "            pin_memory=self.pin_memory,\n            collate_fn=collate_fn,\n",
        "            pin_memory=self.pin_memory,\n"
        "            persistent_workers=self.num_workers > 0,\n"
        "            prefetch_factor=4 if self.num_workers > 0 else None,\n"
        "            collate_fn=collate_fn,\n",
    )
    datamodule_path.write_text(datamodule, encoding="utf-8")

    pl_module_path = code_dir / "src/model/pl_modules.py"
    pl_module = pl_module_path.read_text(encoding="utf-8")
    pl_module = replace_once(
        pl_module,
        "        #assert len(param_dict.keys() - union_params) == 0\n\n"
        "        optim_groups = [\n",
        "        # Train Mamba's A/D/gate parameters without changing baseline groups.\n"
        "        mamba_unclassified = {\n"
        "            name for name in param_dict.keys() - union_params\n"
        "            if 'mamba_encoder' in name\n"
        "        }\n"
        "        no_decay.update(mamba_unclassified)\n"
        "        missing_mamba = {\n"
        "            name for name in param_dict\n"
        "            if 'mamba_encoder' in name and name not in (decay | no_decay)\n"
        "        }\n"
        "        assert not missing_mamba, f'Mamba parameters missing from optimizer: {missing_mamba}'\n\n"
        "        optim_groups = [\n",
        "Mamba optimizer classification",
    )
    pl_module_path.write_text(pl_module, encoding="utf-8")

    train_path = code_dir / "train.py"
    train = train_path.read_text(encoding="utf-8")
    if "import torch\n" not in train:
        train = train.replace("import os\n", "import os\nimport torch\n")
    train = replace_once(
        train,
        "    model = instantiate(cfg.model.pl_module)\n",
        "    model = instantiate(cfg.model.pl_module)\n"
        "    mamba_params = [\n"
        "        (name, parameter) for name, parameter in model.named_parameters()\n"
        "        if 'mamba_encoder' in name\n"
        "    ]\n"
        "    if not mamba_params:\n"
        "        raise RuntimeError('Mamba is not active: no mamba_encoder parameters found')\n"
        "    from src.model.layers.mamba_encoder import fused_cuda_available\n"
        "    if not fused_cuda_available():\n"
        "        raise RuntimeError('Official fused CUDA Mamba path is unavailable')\n"
        "    mamba_count = sum(parameter.numel() for _, parameter in mamba_params)\n"
        "    print(f'MAMBA_ACTIVE=True FUSED_MAMBA_CUDA_ACTIVE=True '"
        "          f'MAMBA_PARAMETER_TENSORS={len(mamba_params)} '"
        "          f'MAMBA_PARAMETERS={mamba_count}')\n"
        "    with open(os.path.join(output_dir, 'mamba_parameters.txt'), 'w') as handle:\n"
        "        handle.write(f'MAMBA_PARAMETERS={mamba_count}\\n')\n"
        "        for name, parameter in mamba_params:\n"
        "            handle.write(f'{name} {tuple(parameter.shape)} {parameter.numel()}\\n')\n",
        "Mamba runtime assertion",
    )
    train = train.replace(
        "    trainer.validate(model, datamodule.val_dataloader())\n",
        "    trainer.validate(model, datamodule=datamodule, ckpt_path='best')\n",
    )
    train = train.replace(
        "def main(cfg):\n",
        "def main(cfg):\n    torch.set_float32_matmul_precision('high')\n",
    )
    train_path.write_text(train, encoding="utf-8")

    verify_path = experiment_root / "verify_mamba_checkpoint.py"
    verify_path.write_text(VERIFY_SOURCE, encoding="utf-8")
    make_executable(verify_path)

    install_path = experiment_root / "install_fused_mamba.sh"
    write_lf(
        install_path,
        INSTALL_SCRIPT_TEMPLATE.format(
            base=base,
            experiment_root=experiment_root,
        ),
    )
    make_executable(install_path)

    run_path = experiment_root / "run_av2_mamba_fused_4gpu.sh"
    write_lf(
        run_path,
        RUN_SCRIPT_TEMPLATE.format(
            base=base,
            experiment_root=experiment_root,
            results_root=results_root,
        ),
    )
    make_executable(run_path)

    manifest = experiment_root / "EXPERIMENT.txt"
    manifest.write_text(
        "SHARP AV2 with official fused CUDA Mamba scene encoder\n"
        "Training: article-aligned 80 epochs, global batch 32, 13 warm-up epochs, "
        "LR 1e-4 to 1e-5\n"
        f"Source code: {source_code}\n"
        f"Experiment code: {code_dir}\n"
        f"Results: {results_root}\n"
        "Placement: after positional token embedding, before dual training and "
        "instance-aware context streaming\n"
        "Mamba: official fused CUDA Mamba-1, depth=1, d_state=16, d_conv=4, expand=2, shared bidirectional scan\n",
        encoding="utf-8",
    )
    (base / "Codes/LATEST_SHARP_AV2_MAMBA_FUSED80.txt").write_text(
        str(experiment_root) + "\n",
        encoding="utf-8",
    )
    (base / "Results/LATEST_SHARP_AV2_MAMBA_FUSED80.txt").write_text(
        str(results_root) + "\n",
        encoding="utf-8",
    )

    print("SETUP_COMPLETE")
    print(f"EXPERIMENT_ROOT={experiment_root}")
    print(f"RESULTS_ROOT={results_root}")
    print(f"INSTALL_COMMAND={install_path}")
    print(f"RUN_COMMAND={run_path}")


if __name__ == "__main__":
    main()
