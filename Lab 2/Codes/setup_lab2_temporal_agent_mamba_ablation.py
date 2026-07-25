#!/usr/bin/env python3
"""Create controlled SHARP baseline and temporal-agent Mamba AV2 experiments."""

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

import causal_conv1d_cuda
import selective_scan_cuda
import mamba_ssm.modules.mamba_simple as mamba_simple
from mamba_ssm.modules.mamba_simple import Mamba


def fused_cuda_available() -> bool:
    return (
        mamba_simple.causal_conv1d_fn is not None
        and mamba_simple.selective_scan_fn is not None
        and hasattr(causal_conv1d_cuda, "causal_conv1d_fwd")
        and hasattr(selective_scan_cuda, "fwd")
    )


class TemporalAgentMamba(nn.Module):
    """Gated bidirectional Mamba over each agent's chronological observations."""

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        dropout: float = 0.1,
        layer_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        if not fused_cuda_available():
            raise RuntimeError("Official split fused CUDA Mamba kernels are unavailable")

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
        lengths = compact_valid.sum(dim=1)

        forward_out = self.forward_mamba(compact)
        backward_input, reverse_index = self._reverse_valid_prefix(
            compact,
            lengths,
            compact_valid,
        )
        backward_reversed = self.backward_mamba(backward_input)
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
    parser.add_argument("--expect", choices=("absent", "temporal"), required=True)
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

    if args.expect == "absent":
        if keys:
            raise SystemExit("ERROR: baseline checkpoint unexpectedly contains Mamba")
        print("BASELINE_CONTROL_VERIFIED=True")
        return

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


RUNNER_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail

LAB_BASE="{base}"
EXPERIMENT_ROOT="{experiment_root}"
CODE_DIR="{code_dir}"
RESULTS_ROOT="{results_root}"
VARIANT="{variant}"
EXPECT_MAMBA="{expect_mamba}"
POINTER_NAME="{pointer_name}"
FUSED_PACKAGES="{fused_packages}"
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
export PYTHONPATH="$FUSED_PACKAGES:$CODE_DIR:${{PYTHONPATH:-}}"
export PYTHONUNBUFFERED=1
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

if [ "$EXPECT_MAMBA" = "temporal" ]; then
  test -f "$FUSED_PACKAGES/selective_scan_cuda.cpython-311-x86_64-linux-gnu.so"
  test -f "$FUSED_PACKAGES/causal_conv1d_cuda.cpython-311-x86_64-linux-gnu.so"
  "$PYTHON_BIN" - <<'PY'
import torch
from src.model.layers.temporal_agent_mamba import (
    TemporalAgentMamba,
    fused_cuda_available,
)

if not fused_cuda_available():
    raise SystemExit("Fused CUDA Mamba is unavailable")

device = torch.device("cuda:0")
module = TemporalAgentMamba(
    dim=128,
    d_state=8,
    d_conv=3,
    expand=1,
    dropout=0.1,
    layer_scale_init=0.01,
).to(device)
x = torch.randn(32, 10, 128, device=device, requires_grad=True)
valid = torch.ones(32, 10, dtype=torch.bool, device=device)
valid[::3, :4] = False
valid[1::3, :7] = False
y = module(x, valid)
if y.shape != x.shape or not torch.isfinite(y).all():
    raise SystemExit("Temporal-agent Mamba smoke test failed")
y.square().mean().backward()
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
PY
fi

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
checkpoint_monitor=minADE6
source_code=$CODE_DIR
dataset=$DATA_DIR
EOF

echo "VARIANT=$VARIANT"
echo "Starting controlled SHARP AV2 experiment on four GPUs"
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
  "$BEST_CHECKPOINT" --expect "$EXPECT_MAMBA" \
  | tee "$RESULTS_DIR/checkpoint_verification.txt"

echo "SHARP_ABLATION_RUN_COMPLETE=True" | tee "$RESULTS_DIR/COMPLETED.txt"
echo "VARIANT=$VARIANT" | tee -a "$RESULTS_DIR/COMPLETED.txt"
echo "RESULTS_DIR=$RESULTS_DIR" | tee -a "$RESULTS_DIR/COMPLETED.txt"
echo "BEST_CHECKPOINT=$BEST_CHECKPOINT" | tee -a "$RESULTS_DIR/COMPLETED.txt"
'''


COMPARE_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path


BASE = Path("/home/server00/M")
POINTERS = {
    "baseline_control": BASE / "Results/LATEST_SHARP_AV2_BASELINE_CONTROL80_RUN.txt",
    "temporal_agent_mamba": BASE / "Results/LATEST_SHARP_AV2_TEMPORAL_MAMBA80_RUN.txt",
}
METRICS = ("MR", "b-minFDE6", "minADE1", "minADE6", "minFDE1", "minFDE6")


def read_metrics(run: Path) -> dict[str, str]:
    log = run / "full_run.log"
    text = log.read_text(errors="replace").replace("\r", "\n")
    values: dict[str, str] = {}
    for metric in METRICS:
        matches = re.findall(
            rf"(?:\||\u2502)\s*{re.escape(metric)}\s*(?:\||\u2502)\s*([0-9.eE+-]+)",
            text,
        )
        if matches:
            values[metric] = matches[-1]
    return values


rows = []
for variant, pointer in POINTERS.items():
    if not pointer.is_file():
        print(f"SKIP {variant}: pointer not found: {pointer}")
        continue
    run = Path(pointer.read_text().strip())
    if not run.joinpath("COMPLETED.txt").is_file():
        print(f"SKIP {variant}: run has not completed: {run}")
        continue
    metrics = read_metrics(run)
    rows.append({"variant": variant, "run": str(run), **metrics})

if not rows:
    raise SystemExit("No completed runs are available")

output = BASE / "Results/SHARP_AV2_TEMPORAL_MAMBA_COMPARISON.csv"
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=("variant", "run", *METRICS))
    writer.writeheader()
    writer.writerows(rows)

print(output)
for row in rows:
    print(row)
'''


SEQUENTIAL_RUNNER_SOURCE = r'''#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "Running exact no-Mamba control first."
bash "$ROOT/run_baseline_control_4gpu.sh"

echo "Control completed. Running temporal-agent Mamba."
bash "$ROOT/run_temporal_agent_mamba_4gpu.sh"

echo "Both controlled experiments completed."
"$ROOT/compare_completed_runs.py"
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
            and any(candidate.glob("causal_conv1d_cuda*.so"))
        ):
            return candidate.resolve()
    raise SystemExit(
        "No verified fused Mamba package directory was found. "
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
        "    trainer.validate(model, datamodule=datamodule, ckpt_path='best')\n",
    )
    train_path.write_text(train, encoding="utf-8")


def patch_baseline(code_dir: Path) -> None:
    sharp_path = code_dir / "src/model/sharp.py"
    sharp = sharp_path.read_text(encoding="utf-8")
    if "mamba" in sharp.lower():
        raise RuntimeError(
            "The baseline source already contains Mamba. "
            "Use the untouched /home/server00/M/Codes/SHARP/Code source."
        )

    train_path = code_dir / "train.py"
    train = train_path.read_text(encoding="utf-8")
    train = replace_once(
        train,
        "    model = instantiate(cfg.model.pl_module)\n",
        "    model = instantiate(cfg.model.pl_module)\n"
        "    unexpected_mamba = [\n"
        "        name for name, _ in model.named_parameters()\n"
        "        if 'mamba' in name.lower()\n"
        "    ]\n"
        "    if unexpected_mamba:\n"
        "        raise RuntimeError(f'Baseline unexpectedly contains Mamba: {unexpected_mamba[:10]}')\n"
        "    print('EXPERIMENT_VARIANT=baseline_control MAMBA_ACTIVE=False')\n",
        "baseline runtime assertion",
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
        "        temporal_mamba_layer_scale=0.01\n",
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
        "    temporal_mamba_layer_scale: 0.01\n",
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
        "    from src.model.layers.temporal_agent_mamba import fused_cuda_available\n"
        "    if not fused_cuda_available():\n"
        "        raise RuntimeError('Official split fused CUDA Mamba path is unavailable')\n"
        "    mamba_count = sum(parameter.numel() for _, parameter in temporal_mamba_params)\n"
        "    print(\n"
        "        'EXPERIMENT_VARIANT=temporal_agent_mamba MAMBA_ACTIVE=True '\n"
        "        f'TEMPORAL_MAMBA_PARAMETER_TENSORS={len(temporal_mamba_params)} '\n"
        "        f'TEMPORAL_MAMBA_PARAMETERS={mamba_count}'\n"
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
    experiment_root = base / "Codes" / f"SHARP_AV2_TEMPORAL_MAMBA_ABLATION_{stamp}"
    results_root = base / "Results" / f"SHARP_AV2_TEMPORAL_MAMBA_ABLATION_{stamp}"
    baseline_code = experiment_root / "baseline_control/Code"
    temporal_code = experiment_root / "temporal_agent_mamba/Code"

    experiment_root.mkdir(parents=True, exist_ok=False)
    results_root.mkdir(parents=True, exist_ok=False)
    for destination in (baseline_code, temporal_code):
        shutil.copytree(
            source_code,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".hydra", "outputs"),
        )
        patch_common(destination)

    patch_baseline(baseline_code)
    patch_temporal_mamba(temporal_code)

    dependency_link = experiment_root / "fused_packages"
    os.symlink(fused_packages, dependency_link, target_is_directory=True)
    (experiment_root / "FUSED_PACKAGE_SOURCE.txt").write_text(
        str(fused_packages) + "\n",
        encoding="utf-8",
    )

    verify_path = experiment_root / "verify_checkpoint.py"
    write_lf(verify_path, VERIFY_SOURCE)
    make_executable(verify_path)

    baseline_runner = experiment_root / "run_baseline_control_4gpu.sh"
    write_lf(
        baseline_runner,
        RUNNER_TEMPLATE.format(
            base=base,
            experiment_root=experiment_root,
            code_dir=baseline_code,
            results_root=results_root,
            variant="baseline_control",
            expect_mamba="absent",
            pointer_name="LATEST_SHARP_AV2_BASELINE_CONTROL80_RUN.txt",
            fused_packages=dependency_link,
        ),
    )
    make_executable(baseline_runner)

    temporal_runner = experiment_root / "run_temporal_agent_mamba_4gpu.sh"
    write_lf(
        temporal_runner,
        RUNNER_TEMPLATE.format(
            base=base,
            experiment_root=experiment_root,
            code_dir=temporal_code,
            results_root=results_root,
            variant="temporal_agent_mamba",
            expect_mamba="temporal",
            pointer_name="LATEST_SHARP_AV2_TEMPORAL_MAMBA80_RUN.txt",
            fused_packages=dependency_link,
        ),
    )
    make_executable(temporal_runner)

    sequential_runner = experiment_root / "run_control_then_temporal.sh"
    write_lf(sequential_runner, SEQUENTIAL_RUNNER_SOURCE)
    make_executable(sequential_runner)

    compare_path = experiment_root / "compare_completed_runs.py"
    write_lf(compare_path, COMPARE_SOURCE)
    make_executable(compare_path)

    manifest = experiment_root / "EXPERIMENT.txt"
    manifest.write_text(
        "Controlled SHARP AV2 temporal-agent Mamba ablation\n"
        f"Original source: {source_code}\n"
        f"Baseline code: {baseline_code}\n"
        f"Temporal Mamba code: {temporal_code}\n"
        f"Results root: {results_root}\n"
        f"Reused fused packages (read-only dependency): {fused_packages}\n"
        "Shared settings: seed 2333, 80 epochs, 4 GPUs, batch 8/GPU, "
        "global batch 32, 6 workers/process, SyncBatchNorm, AdamW, "
        "LR 1e-4 to 1e-5, warmup ratio 0.167, top-3 minADE6 checkpoints.\n"
        "Baseline: untouched SHARP architecture; runtime assertion requires zero Mamba parameters.\n"
        "Temporal variant: original four agent-history attention blocks retained; "
        "one bidirectional Mamba block inserted between temporal blocks 2 and 3. "
        "Separate directions, d_state=8, d_conv=3, expand=1, dropout=0.1, "
        "per-channel LayerScale=0.01, valid observations compacted chronologically.\n"
        "No previous code, result, checkpoint, or dependency is modified or deleted.\n",
        encoding="utf-8",
    )

    (base / "Codes/LATEST_SHARP_AV2_TEMPORAL_MAMBA_ABLATION.txt").write_text(
        str(experiment_root) + "\n",
        encoding="utf-8",
    )
    (base / "Results/LATEST_SHARP_AV2_TEMPORAL_MAMBA_ABLATION.txt").write_text(
        str(results_root) + "\n",
        encoding="utf-8",
    )

    print("SETUP_COMPLETE")
    print(f"EXPERIMENT_ROOT={experiment_root}")
    print(f"RESULTS_ROOT={results_root}")
    print(f"BASELINE_RUNNER={baseline_runner}")
    print(f"TEMPORAL_MAMBA_RUNNER={temporal_runner}")
    print(f"SEQUENTIAL_RUNNER={sequential_runner}")


if __name__ == "__main__":
    main()
