#!/usr/bin/env python3
"""Create the rotation-safe stable-CUDA SHARP temporal-agent Mamba AV2 experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import stat
from pathlib import Path


TEMPORAL_MAMBA_SOURCE = r'''"""Temporal-agent Mamba for chronological SHARP agent histories."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import selective_scan_cuda
import mamba_ssm.modules.mamba_simple as mamba_simple
from mamba_ssm.modules.mamba_simple import Mamba


def configure_stable_cuda_backend() -> None:
    """Use CUDA/cuDNN Conv1d and retain the fused CUDA selective scan.

    The custom causal-conv extension corrupted CUDA state after repeated
    temporal-agent calls on RTX 2080 Ti. Mamba's documented slow path already
    supports nn.Conv1d when causal_conv1d_fn is unavailable, so disable only
    that optional extension for this isolated experiment.
    """
    mamba_simple.causal_conv1d_fn = None
    mamba_simple.causal_conv1d_update = None


configure_stable_cuda_backend()


def stable_cuda_available() -> bool:
    return (
        mamba_simple.causal_conv1d_fn is None
        and mamba_simple.selective_scan_fn is not None
        and hasattr(selective_scan_cuda, "fwd")
    )


def backend_summary() -> str:
    return (
        "CAUSAL_CONV_BACKEND=torch_cuda_cudnn_conv1d "
        "SELECTIVE_SCAN_BACKEND=fused_selective_scan_cuda"
    )


class TemporalAgentMamba(nn.Module):
    """Gated bidirectional Mamba over chronological agent observations.

    Fixed-size contiguous batches feed CUDA/cuDNN Conv1d and the official fused
    selective-scan kernel. Chunking along the agent dimension is mathematically
    independent and does not change sequence semantics or learned parameters.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        dropout: float = 0.1,
        layer_scale_init: float = 0.01,
        agent_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if not stable_cuda_available():
            raise RuntimeError("Stable CUDA Mamba backend is unavailable")

        self.norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.backward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.direction_logits = nn.Parameter(torch.zeros(dim))
        self.layer_scale = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.dropout = nn.Dropout(dropout)
        if agent_chunk_size < 1:
            raise ValueError("agent_chunk_size must be positive")
        self.agent_chunk_size = agent_chunk_size

    @staticmethod
    def _compact_valid(
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move valid observations to a prefix while preserving chronological order."""
        batch, length, _ = x.shape
        compact_rank = valid_mask.long().cumsum(dim=1) - 1
        row_index = torch.arange(batch, device=x.device).unsqueeze(1).expand(batch, length)
        compact = torch.zeros_like(x)
        compact[row_index[valid_mask], compact_rank[valid_mask]] = x[valid_mask]
        lengths = valid_mask.sum(dim=1)
        compact_valid = (
            torch.arange(length, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        )
        return compact, compact_rank, compact_valid

    @staticmethod
    def _reverse_valid_prefix(
        x: torch.Tensor,
        lengths: torch.Tensor,
        valid_prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reverse only each valid prefix, leaving padding after the sequence."""
        _, length, dim = x.shape
        positions = torch.arange(length, device=x.device).unsqueeze(0)
        reverse_index = (lengths.unsqueeze(1) - 1 - positions).clamp(0, length - 1)
        gather_index = reverse_index.unsqueeze(-1).expand(-1, -1, dim)
        reversed_x = x.gather(1, gather_index)
        reversed_x = reversed_x * valid_prefix.unsqueeze(-1).to(dtype=x.dtype)
        return reversed_x, gather_index

    def _run_fixed_chunks(
        self,
        module: Mamba,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run independent agent sequences in fixed, contiguous CUDA batches."""
        if x.shape[0] == 0:
            return x

        outputs = []
        for start in range(0, x.shape[0], self.agent_chunk_size):
            chunk = x[start : start + self.agent_chunk_size].contiguous()
            real_rows = chunk.shape[0]
            if real_rows < self.agent_chunk_size:
                chunk = F.pad(
                    chunk,
                    (0, 0, 0, 0, 0, self.agent_chunk_size - real_rows),
                )
            outputs.append(module(chunk.contiguous())[:real_rows])
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3 or valid_mask.shape != x.shape[:2]:
            raise ValueError(
                f"Expected x [agents,time,dim] and matching mask, got "
                f"{tuple(x.shape)} and {tuple(valid_mask.shape)}"
            )

        normalized = self.norm(x)
        compact, compact_rank, compact_valid = self._compact_valid(
            normalized,
            valid_mask,
        )
        compact = compact.contiguous()
        lengths = compact_valid.sum(dim=1)

        forward_out = self._run_fixed_chunks(self.forward_mamba, compact)
        backward_input, reverse_index = self._reverse_valid_prefix(
            compact,
            lengths,
            compact_valid,
        )
        backward_reversed = self._run_fixed_chunks(
            self.backward_mamba,
            backward_input.contiguous(),
        )
        backward_out = backward_reversed.gather(1, reverse_index)

        direction_gate = torch.sigmoid(self.direction_logits).view(1, 1, -1)
        compact_update = (
            direction_gate * forward_out
            + (1.0 - direction_gate) * backward_out
        )
        compact_update = compact_update * compact_valid.unsqueeze(-1).to(x.dtype)

        batch, length, _ = x.shape
        row_index = torch.arange(batch, device=x.device).unsqueeze(1).expand(
            batch,
            length,
        )
        update = torch.zeros_like(x)
        update[valid_mask] = compact_update[
            row_index[valid_mask],
            compact_rank[valid_mask],
        ]

        return x + self.layer_scale.view(1, 1, -1) * self.dropout(update)
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
    keys = [key for key in state_dict if "mamba" in key.lower()]
    temporal_keys = [key for key in keys if "temporal_agent_mamba" in key]
    count = sum(state_dict[key].numel() for key in temporal_keys)

    print(f"CHECKPOINT={args.checkpoint}")
    print(f"ALL_MAMBA_KEYS={len(keys)}")
    print(f"TEMPORAL_MAMBA_KEYS={len(temporal_keys)}")
    print(f"TEMPORAL_MAMBA_PARAMETERS={count}")

    if not temporal_keys or count == 0:
        raise SystemExit("ERROR: temporal-agent Mamba parameters are missing")
    unrelated = [key for key in keys if key not in temporal_keys]
    if unrelated:
        raise SystemExit(f"ERROR: unexpected non-temporal Mamba keys: {unrelated[:10]}")

    for key in temporal_keys[:30]:
        print(key)
    print("TEMPORAL_AGENT_MAMBA_VERIFIED=True")


if __name__ == "__main__":
    main()
'''

RUNTIME_PATCH_SOURCE = r'''import warnings

import numpy as np

for _name, _value in (("bool", bool), ("int", int), ("float", float)):
    if _name not in np.__dict__:
        setattr(np, _name, _value)

# Suppress only compatibility notices verified as harmless for this pinned stack.
warnings.filterwarnings(
    "ignore",
    message=r"The 'repr' attribute with value False was provided.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"The 'frozen' attribute with value True was provided.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"No device id is provided via .*",
)
'''

RUNNER_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail

LAB_BASE="{base}"
EXPERIMENT_ROOT="{experiment_root}"
CODE_DIR="{code_dir}"
RESULTS_ROOT="{results_root}"
VARIANT="{variant}"
POINTER_NAME="{pointer_name}"
FUSED_PACKAGES="{fused_packages}"
RUNTIME_PATCH="{runtime_patch}"
ATTEMPT_ID=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="$RESULTS_ROOT/$VARIANT/$ATTEMPT_ID"
DATA_DIR="$LAB_BASE/Datasets/AV2/sharp_processed"
PYTHON_BIN="$LAB_BASE/Codes/envs/sharp/bin/python"

# These values match the completed Lab 2 scene-Mamba experiment.
BATCH_SIZE=8
WORKERS=6

mkdir -p "$RESULTS_DIR"
printf '%s\n' "$RESULTS_DIR" > "$LAB_BASE/Results/$POINTER_NAME"

source "$LAB_BASE/Codes/miniforge3/etc/profile.d/conda.sh"
conda activate "$LAB_BASE/Codes/envs/sharp"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$RUNTIME_PATCH:$FUSED_PACKAGES:$CODE_DIR:${{PYTHONPATH:-}}"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export NO_COLOR=1
export RICH_NO_COLOR=1
export EXPERIMENT_VARIANT="$VARIANT"
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
test -x "$PYTHON_BIN"

if pgrep -u "$USER" -f "$PYTHON_BIN.*train.py" >/dev/null 2>&1; then
  echo "ERROR: another SHARP training process is already running."
  echo "Run only one four-GPU experiment at a time."
  ps -u "$USER" -o pid,etime,%cpu,%mem,cmd | grep -E "train.py" | grep -v grep || true
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected four CUDA GPUs, found {{torch.cuda.device_count()}}")
print("CUDA GPUs:")
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

cd "$CODE_DIR"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path

sharp = Path("src/model/sharp.py").read_text()
if "torch.inverse(rot_mat)" in sharp:
    raise SystemExit("Unsafe cuSOLVER rotation inverse is still present")
if "rot_mat.transpose(1, 2)" not in sharp:
    raise SystemExit("Rotation-transpose patch is missing")
for name in (
    "src/model/layers/custom_transformer_blocks.py",
    "src/model/layers/transformer_blocks.py",
):
    layer = Path(name).read_text()
    if "from timm.models.layers" in layer:
        raise SystemExit(f"Deprecated timm import remains in {{name}}")
    if "def _match_attention_mask_dtypes(" not in layer:
        raise SystemExit(f"Attention-mask dtype patch is missing from {{name}}")
from src.model.sharp import Sharp

print("ROTATION_TRANSPOSE_PATCH_ACTIVE=True")
print("ORIGINAL_SHARP_ATTENTION_MASK_COMPATIBILITY_ACTIVE=True")
print("DEPRECATED_TIMM_IMPORTS_PRESENT=False")
print("SHARP_MODEL_IMPORT_OK=True")
PY

# Verify the stable CUDA backend and temporal-agent Mamba before training.
  test -f "$FUSED_PACKAGES/selective_scan_cuda.cpython-311-x86_64-linux-gnu.so"
  "$PYTHON_BIN" - <<'PY'
import torch
from src.model.layers.temporal_agent_mamba import (
    TemporalAgentMamba,
    backend_summary,
    stable_cuda_available,
)

if not stable_cuda_available():
    raise SystemExit("Stable CUDA Mamba is unavailable")

device = torch.device("cuda:0")
module = TemporalAgentMamba(
    dim=128,
    d_state=8,
    d_conv=3,
    expand=1,
    dropout=0.1,
    layer_scale_init=0.01,
    agent_chunk_size=128,
).to(device)
x = torch.randn(257, 10, 128, device=device, requires_grad=True)
valid = torch.ones(257, 10, dtype=torch.bool, device=device)
valid[::3, :4] = False
valid[1::3, :7] = False
y = module(x, valid)
if y.shape != x.shape or not torch.isfinite(y).all():
    raise SystemExit("Temporal-agent Mamba smoke test failed")
y.square().mean().backward()
torch.cuda.synchronize()
grad_count = sum(
    parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
    for parameter in module.parameters()
)
if grad_count == 0:
    raise SystemExit("Temporal-agent Mamba produced no finite gradients")
parameter_count = sum(parameter.numel() for parameter in module.parameters())
print(
    f"TEMPORAL_AGENT_MAMBA_SMOKE_TEST_OK shape={{tuple(y.shape)}} "
    f"parameters={{parameter_count}} gradients={{grad_count}}"
)
print(backend_summary())
PY

echo "Running an exact blocking 256-batch four-GPU DDP preflight on real AV2 data."
PREFLIGHT_DIR="$RESULTS_DIR/preflight"
mkdir -p "$PREFLIGHT_DIR"

SHARP_PREFLIGHT_ONLY=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CUDA_LAUNCH_BLOCKING=1 \
"$PYTHON_BIN" -u train.py \
  seed=2333 \
  gpus=4 \
  epochs=1 \
  batch_size="$BATCH_SIZE" \
  output_dir="$PREFLIGHT_DIR/run" \
  datamodule.pl_module.data_root="$DATA_DIR" \
  datamodule.pl_module.num_workers="$WORKERS" \
  model.pl_module.optim.lr=0.0001 \
  model.pl_module.optim.min_lr=0.00001 \
  model.pl_module.optim.warmup_ratio=0.167 \
  trainer.devices=4 \
  trainer.strategy=ddp_find_unused_parameters_false \
  trainer.sync_batchnorm=true \
  trainer.num_sanity_val_steps=0 \
  +trainer.limit_train_batches=256 \
  +trainer.limit_val_batches=0 \
  callbacks.0.save_top_k=0 \
  2>&1 | tee "$PREFLIGHT_DIR/preflight.log"

printf '%s\n' "REAL_AV2_PREFLIGHT_PASSED=True" \
  > "$PREFLIGHT_DIR/PREFLIGHT_PASSED.txt"
echo "Exact four-GPU AV2 preflight passed beyond the prior failure point. Starting the full run."

cat > "$RESULTS_DIR/RUN_SETTINGS.txt" <<EOF
variant=$VARIANT
seed=2333
epochs=80
gpus=4
batch_per_gpu=$BATCH_SIZE
global_batch=32
workers_per_process=$WORKERS
sync_batchnorm=true
learning_rate=0.0001
minimum_learning_rate=0.00001
warmup_ratio=0.167
weight_decay=0.01
gradient_clip_val=5
agent_chunk_size=128
causal_conv_backend=torch_cuda_cudnn_conv1d
selective_scan_backend=fused_selective_scan_cuda
rotation_inverse_backend=orthogonal_transpose_no_cusolver
attention_mask_dtype_matched=true
compatibility_warnings_filtered=true
real_data_preflight_batches=256
checkpoint_monitor=minADE6
source_code=$CODE_DIR
dataset=$DATA_DIR
EOF

echo "VARIANT=$VARIANT"
echo "Starting SHARP temporal-agent Mamba AV2 experiment on four GPUs"
echo "Per-GPU batch: 8; global batch: 32; workers/process: 6"
echo "Schedule: 80 epochs, warmup_ratio 0.167, LR 1e-4 -> 1e-5"

"$PYTHON_BIN" -u train.py \
  seed=2333 \
  gpus=4 \
  epochs=80 \
  batch_size="$BATCH_SIZE" \
  output_dir="$RESULTS_DIR/run" \
  datamodule.pl_module.data_root="$DATA_DIR" \
  datamodule.pl_module.num_workers="$WORKERS" \
  model.pl_module.optim.lr=0.0001 \
  model.pl_module.optim.min_lr=0.00001 \
  model.pl_module.optim.warmup_ratio=0.167 \
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
    raise SystemExit("No monitored minADE6 checkpoint found")
print(min(candidates, key=lambda item: item[0])[1])
PY
)

printf '%s\n' "$BEST_CHECKPOINT" > "$RESULTS_DIR/BEST_CHECKPOINT.txt"
"$PYTHON_BIN" "$EXPERIMENT_ROOT/verify_checkpoint.py" \
  "$BEST_CHECKPOINT" \
  | tee "$RESULTS_DIR/checkpoint_verification.txt"

echo "SHARP_TEMPORAL_AGENT_MAMBA_ROTATIONFIX_RUN_COMPLETE=True" | tee "$RESULTS_DIR/COMPLETED.txt"
echo "VARIANT=$VARIANT" | tee -a "$RESULTS_DIR/COMPLETED.txt"
echo "RESULTS_DIR=$RESULTS_DIR" | tee -a "$RESULTS_DIR/COMPLETED.txt"
echo "BEST_CHECKPOINT=$BEST_CHECKPOINT" | tee -a "$RESULTS_DIR/COMPLETED.txt"
'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def write_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace("\r\n", "\n"))


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def find_fused_packages(base: Path) -> Path:
    candidates: list[Path] = []
    pointer = base / "Codes/LATEST_SHARP_AV2_MAMBA_FUSED80.txt"
    if pointer.is_file():
        candidates.append(Path(pointer.read_text(encoding="utf-8").strip()) / "fused_packages")
    candidates.extend(
        path / "fused_packages"
        for path in sorted(
            base.joinpath("Codes").glob("SHARP_AV2_MAMBA_FUSED80_*"),
            reverse=True,
        )
    )
    for candidate in candidates:
        if (
            candidate.is_dir()
            and any(candidate.glob("selective_scan_cuda*.so"))
            and candidate.joinpath("mamba_ssm").is_dir()
        ):
            return candidate.resolve()
    raise SystemExit(
        "No verified Mamba package with fused selective scan was found. "
        "The completed SHARP_AV2_MAMBA_FUSED80 experiment must remain available."
    )


def patch_common(code_dir: Path) -> None:
    model_cfg_path = code_dir / "conf/model/Sharp_av2.yaml"
    model_cfg = model_cfg_path.read_text(encoding="utf-8")
    model_cfg = replace_once(
        model_cfg,
        "    lr: 1e-3\n",
        "    lr: 1e-4\n",
        "learning-rate",
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

    attention_mask_helper = '''

def _match_attention_mask_dtypes(
    attn_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
):
    """Return equivalent masks with the dtype contract required by PyTorch."""
    if (
        attn_mask is None
        or key_padding_mask is None
        or attn_mask.dtype == key_padding_mask.dtype
    ):
        return attn_mask, key_padding_mask

    if attn_mask.is_floating_point() and key_padding_mask.dtype == torch.bool:
        float_padding_mask = torch.zeros_like(
            key_padding_mask, dtype=attn_mask.dtype
        )
        key_padding_mask = float_padding_mask.masked_fill(
            key_padding_mask, float("-inf")
        )
        return attn_mask, key_padding_mask

    if key_padding_mask.is_floating_point() and attn_mask.dtype == torch.bool:
        float_attn_mask = torch.zeros_like(
            attn_mask, dtype=key_padding_mask.dtype
        )
        attn_mask = float_attn_mask.masked_fill(attn_mask, float("-inf"))
        return attn_mask, key_padding_mask

    if attn_mask.is_floating_point() and key_padding_mask.is_floating_point():
        return attn_mask, key_padding_mask.to(dtype=attn_mask.dtype)

    raise TypeError(
        "Unsupported attention-mask dtype pair: "
        f"{attn_mask.dtype}, {key_padding_mask.dtype}"
    )
'''

    for layer_name in (
        "custom_transformer_blocks.py",
        "transformer_blocks.py",
    ):
        layer_path = code_dir / "src/model/layers" / layer_name
        layer = layer_path.read_text(encoding="utf-8")
        layer = layer.replace(
            "from timm.models.layers import DropPath",
            "from timm.layers import DropPath",
        )
        if "from timm.models.layers" in layer:
            raise RuntimeError(f"Deprecated timm import remains in {layer_path}")
        layer = replace_once(
            layer,
            "from timm.layers import DropPath\n",
            "from timm.layers import DropPath\n" + attention_mask_helper,
            f"attention-mask helper in {layer_name}",
        )
        attention_calls = 0
        for call in (
            "        src2 = self.attn(\n",
            "        attn_output = self.attn(\n",
        ):
            count = layer.count(call)
            attention_calls += count
            layer = layer.replace(
                call,
                "        mask, key_padding_mask = "
                "_match_attention_mask_dtypes(mask, key_padding_mask)\n"
                + call,
            )
        if attention_calls == 0:
            raise RuntimeError(f"No attention calls patched in {layer_path}")
        layer_path.write_text(layer, encoding="utf-8")

    sharp_path = code_dir / "src/model/sharp.py"
    sharp = sharp_path.read_text(encoding="utf-8")
    sharp = replace_once(
        sharp,
        "torch.inverse(rot_mat)",
        "rot_mat.transpose(1, 2)",
        "orthogonal rotation inverse",
    )
    sharp_path.write_text(sharp, encoding="utf-8")

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

    train_path = code_dir / "train.py"
    train = train_path.read_text(encoding="utf-8")
    if "import torch\n" not in train:
        train = train.replace("import os\n", "import os\nimport torch\n")
    train = train.replace(
        "def main(cfg):\n",
        "def main(cfg):\n    torch.set_float32_matmul_precision('high')\n",
        1,
    )
    train = train.replace(
        "    trainer.validate(model, datamodule.val_dataloader())\n",
        "    if os.environ.get('SHARP_PREFLIGHT_ONLY') != '1':\n"
        "        trainer.validate(model, datamodule=datamodule, ckpt_path='best')\n",
    )
    train_path.write_text(train, encoding="utf-8")


def patch_temporal_mamba(code_dir: Path) -> None:
    layer_path = code_dir / "src/model/layers/temporal_agent_mamba.py"
    layer_path.write_text(TEMPORAL_MAMBA_SOURCE, encoding="utf-8")

    sharp_path = code_dir / "src/model/sharp.py"
    sharp = sharp_path.read_text(encoding="utf-8")
    if "mamba" in sharp.lower():
        raise RuntimeError(
            "The source SHARP model already contains Mamba; refusing to stack modules."
        )
    sharp = replace_once(
        sharp,
        "from .layers.multimodal_decoder_attn import MultimodalDecoder            \n",
        "from .layers.multimodal_decoder_attn import MultimodalDecoder            \n"
        "from .layers.temporal_agent_mamba import TemporalAgentMamba\n",
        "temporal Mamba import",
    )
    sharp = replace_once(
        sharp,
        '        dm="av2",\n        k=6\n',
        '        dm="av2",\n'
        "        k=6,\n"
        "        temporal_mamba_after_block=2,\n"
        "        temporal_mamba_d_state=8,\n"
        "        temporal_mamba_d_conv=3,\n"
        "        temporal_mamba_expand=1,\n"
        "        temporal_mamba_dropout=0.1,\n"
        "        temporal_mamba_layer_scale=0.01,\n"
        "        temporal_mamba_agent_chunk_size=128\n",
        "Sharp constructor",
    )
    sharp = replace_once(
        sharp,
        "        self.initialize_weights()\n        return\n",
        "        self.initialize_weights()\n"
        "        # Construct Mamba after SHARP initialization to preserve official Mamba init.\n"
        "        self.temporal_mamba_after_block = temporal_mamba_after_block\n"
        "        self.temporal_agent_mamba = TemporalAgentMamba(\n"
        "            dim=embed_dim,\n"
        "            d_state=temporal_mamba_d_state,\n"
        "            d_conv=temporal_mamba_d_conv,\n"
        "            expand=temporal_mamba_expand,\n"
        "            dropout=temporal_mamba_dropout,\n"
        "            layer_scale_init=temporal_mamba_layer_scale,\n"
        "            agent_chunk_size=temporal_mamba_agent_chunk_size,\n"
        "        )\n"
        "        return\n",
        "temporal Mamba construction",
    )
    sharp = replace_once(
        sharp,
        "        for blk in self.h_embed:\n"
        "            actor_feat = blk(actor_feat, key_padding_mask=kpm)\n",
        "        for block_index, blk in enumerate(self.h_embed):\n"
        "            actor_feat = blk(actor_feat, key_padding_mask=kpm)\n"
        "            if block_index + 1 == self.temporal_mamba_after_block:\n"
        "                actor_feat = self.temporal_agent_mamba(actor_feat, ~kpm)\n",
        "temporal Mamba forward placement",
    )
    sharp_path.write_text(sharp, encoding="utf-8")

    model_cfg_path = code_dir / "conf/model/Sharp_av2.yaml"
    model_cfg = model_cfg_path.read_text(encoding="utf-8")
    model_cfg = replace_once(
        model_cfg,
        "    drop_path: 0.2\n",
        "    drop_path: 0.2\n"
        "    temporal_mamba_after_block: 2\n"
        "    temporal_mamba_d_state: 8\n"
        "    temporal_mamba_d_conv: 3\n"
        "    temporal_mamba_expand: 1\n"
        "    temporal_mamba_dropout: 0.1\n"
        "    temporal_mamba_layer_scale: 0.01\n"
        "    temporal_mamba_agent_chunk_size: 128\n",
        "temporal Mamba model configuration",
    )
    model_cfg_path.write_text(model_cfg, encoding="utf-8")

    pl_module_path = code_dir / "src/model/pl_modules.py"
    pl_module = pl_module_path.read_text(encoding="utf-8")
    pl_module = replace_once(
        pl_module,
        "        #assert len(param_dict.keys() - union_params) == 0\n\n"
        "        optim_groups = [\n",
        "        # Include every temporal Mamba parameter in the existing AdamW groups.\n"
        "        temporal_mamba_unclassified = {\n"
        "            name for name in param_dict.keys() - union_params\n"
        "            if 'temporal_agent_mamba' in name\n"
        "        }\n"
        "        no_decay.update(temporal_mamba_unclassified)\n"
        "        missing_temporal_mamba = {\n"
        "            name for name in param_dict\n"
        "            if 'temporal_agent_mamba' in name and name not in (decay | no_decay)\n"
        "        }\n"
        "        assert not missing_temporal_mamba, (\n"
        "            f'Temporal Mamba parameters missing from optimizer: {missing_temporal_mamba}'\n"
        "        )\n\n"
        "        optim_groups = [\n",
        "temporal Mamba optimizer classification",
    )
    pl_module_path.write_text(pl_module, encoding="utf-8")

    train_path = code_dir / "train.py"
    train = train_path.read_text(encoding="utf-8")
    train = replace_once(
        train,
        "    model = instantiate(cfg.model.pl_module)\n",
        "    model = instantiate(cfg.model.pl_module)\n"
        "    temporal_mamba_params = [\n"
        "        (name, parameter) for name, parameter in model.named_parameters()\n"
        "        if 'temporal_agent_mamba' in name\n"
        "    ]\n"
        "    unrelated_mamba = [\n"
        "        name for name, _ in model.named_parameters()\n"
        "        if 'mamba' in name.lower() and 'temporal_agent_mamba' not in name\n"
        "    ]\n"
        "    if not temporal_mamba_params:\n"
        "        raise RuntimeError('Temporal-agent Mamba is not active')\n"
        "    if unrelated_mamba:\n"
        "        raise RuntimeError(f'Unexpected Mamba modules are active: {unrelated_mamba[:10]}')\n"
        "    from src.model.layers.temporal_agent_mamba import backend_summary, stable_cuda_available\n"
        "    if not stable_cuda_available():\n"
        "        raise RuntimeError('Stable CUDA Mamba path is unavailable')\n"
        "    mamba_count = sum(parameter.numel() for _, parameter in temporal_mamba_params)\n"
        "    print(\n"
        "        'EXPERIMENT_VARIANT=temporal_agent_mamba_rotationfix MAMBA_ACTIVE=True '\n"
        "        f'TEMPORAL_MAMBA_PARAMETER_TENSORS={len(temporal_mamba_params)} '\n"
        "        f'TEMPORAL_MAMBA_PARAMETERS={mamba_count} AGENT_CHUNK_SIZE=128 '\n"
        "        f'{backend_summary()}'\n"
        "    )\n"
        "    with open(os.path.join(output_dir, 'temporal_mamba_parameters.txt'), 'w') as handle:\n"
        "        handle.write(f'TEMPORAL_MAMBA_PARAMETERS={mamba_count}\\n')\n"
        "        for name, parameter in temporal_mamba_params:\n"
        "            handle.write(f'{name} {tuple(parameter.shape)} {parameter.numel()}\\n')\n",
        "temporal Mamba runtime assertion",
    )
    train_path.write_text(train, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/home/server00/M")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    source_code = base / "Codes/SHARP/Code"
    dataset = base / "Datasets/AV2/sharp_processed"
    python_bin = base / "Codes/envs/sharp/bin/python"

    if not source_code.is_dir():
        raise SystemExit(f"Missing original SHARP source: {source_code}")
    if not dataset.joinpath("train").is_dir() or not dataset.joinpath("val").is_dir():
        raise SystemExit(f"Missing processed AV2 data: {dataset}")
    if not python_bin.is_file():
        raise SystemExit(f"Missing SHARP Python environment: {python_bin}")

    fused_packages = find_fused_packages(base)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_root = base / "Codes" / f"SHARP_AV2_TEMPORAL_AGENT_MAMBA_ROTATIONFIX80_{stamp}"
    results_root = base / "Results" / f"SHARP_AV2_TEMPORAL_AGENT_MAMBA_ROTATIONFIX80_{stamp}"
    code_dir = experiment_root / "Code"

    experiment_root.mkdir(parents=True, exist_ok=False)
    results_root.mkdir(parents=True, exist_ok=False)
    runtime_patch = experiment_root / "runtime_patch"
    runtime_patch.mkdir(parents=True, exist_ok=False)
    write_lf(runtime_patch / "sitecustomize.py", RUNTIME_PATCH_SOURCE)
    shutil.copytree(
        source_code,
        code_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".hydra", "outputs"),
    )
    patch_common(code_dir)
    patch_temporal_mamba(code_dir)

    dependency_link = experiment_root / "fused_packages"
    os.symlink(fused_packages, dependency_link, target_is_directory=True)
    (experiment_root / "FUSED_PACKAGE_SOURCE.txt").write_text(
        str(fused_packages) + "\n",
        encoding="utf-8",
    )

    verify_path = experiment_root / "verify_checkpoint.py"
    write_lf(verify_path, VERIFY_SOURCE)
    make_executable(verify_path)

    runner = experiment_root / "run_temporal_agent_mamba_rotationfix_4gpu.sh"
    write_lf(
        runner,
        RUNNER_TEMPLATE.format(
            base=base,
            experiment_root=experiment_root,
            code_dir=code_dir,
            results_root=results_root,
            variant="temporal_agent_mamba_rotationfix",
            pointer_name="LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_ROTATIONFIX80_RUN.txt",
            fused_packages=dependency_link,
            runtime_patch=runtime_patch,
        ),
    )
    make_executable(runner)

    manifest = experiment_root / "EXPERIMENT.txt"
    manifest.write_text(
        "SHARP AV2 rotation-safe stable-CUDA temporal-agent Mamba experiment\n"
        f"Original source: {source_code}\n"
        f"Experiment code: {code_dir}\n"
        f"Results root: {results_root}\n"
        f"Reused selective-scan package (read-only dependency): {fused_packages}\n"
        "Training/runtime settings match the completed Lab 2 scene-Mamba run: "
        "seed 2333, 80 epochs, 4-GPU DDP, batch 8/GPU, global batch 32, "
        "6 workers/process, SyncBatchNorm, AdamW, LR 1e-4 to 1e-5, "
        "warmup ratio 0.167, top-3 minADE6 checkpoints plus last.ckpt.\n"
        "Architecture change: the previous scene-token Mamba is absent. "
        "The original four SHARP agent-history attention blocks are retained, "
        "and one bidirectional temporal-agent Mamba block is inserted between "
        "blocks 2 and 3. It uses separate directions, d_state=8, d_conv=3, "
        "expand=1, dropout=0.1, per-channel LayerScale=0.01, fixed agent chunks of 128, contiguous CUDA inputs, CUDA/cuDNN nn.Conv1d instead of the unstable optional causal-conv extension, fused CUDA selective scan, and chronologically "
        "compacted valid observations.\n"
        "Numerical runtime fix: because rot_mat is orthogonal, its exact inverse is computed as transpose without invoking cuSOLVER. Original SHARP attention mask dtypes are matched without changing mask values or semantics; no attention type is replaced. No other SHARP architecture component is replaced. No previous code, "
        "result, checkpoint, or dependency is modified or deleted.\n",
        encoding="utf-8",
    )

    (base / "Codes/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_ROTATIONFIX80.txt").write_text(
        str(experiment_root) + "\n",
        encoding="utf-8",
    )
    (base / "Results/LATEST_SHARP_AV2_TEMPORAL_AGENT_MAMBA_ROTATIONFIX80.txt").write_text(
        str(results_root) + "\n",
        encoding="utf-8",
    )

    print("SETUP_COMPLETE")
    print(f"EXPERIMENT_ROOT={experiment_root}")
    print(f"RESULTS_ROOT={results_root}")
    print(f"RUN_COMMAND={runner}")


if __name__ == "__main__":
    main()
