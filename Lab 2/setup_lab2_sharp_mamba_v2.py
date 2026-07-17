#!/usr/bin/env python3
"""Create an isolated SHARP + native PyTorch Mamba experiment on Lab 2."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import stat
from pathlib import Path


MAMBA_ENCODER_SOURCE = r'''"""Native PyTorch selective state-space encoder for SHARP scene tokens.

This follows the main Mamba design: gated input projection, causal depthwise
convolution, input-dependent delta/B/C parameters, a stable selective SSM
recurrence, skip connection, and output projection.  It avoids external CUDA
extensions so it works on Lab 2 without nvcc.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveMambaSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(a.log().repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Stable Mamba-style delta initialization in [0.001, 0.1].
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        inverse_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        b_param: torch.Tensor,
        c_param: torch.Tensor,
    ) -> torch.Tensor:
        # Keep the recurrent state in FP32 for numerical stability.
        input_dtype = u.dtype
        u = u.float()
        delta = delta.float()
        b_param = b_param.float()
        c_param = c_param.float()
        a = -torch.exp(self.A_log.float())
        d = self.D.float()

        batch, length, _ = u.shape
        state = torch.zeros(
            batch,
            self.d_inner,
            self.d_state,
            device=u.device,
            dtype=torch.float32,
        )
        outputs = []

        for index in range(length):
            delta_t = delta[:, index]
            u_t = u[:, index]
            b_t = b_param[:, index]
            c_t = c_param[:, index]

            delta_a = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
            delta_b = delta_t.unsqueeze(-1) * b_t.unsqueeze(1)
            state = delta_a * state + delta_b * u_t.unsqueeze(-1)
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + d * u_t
            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :length].transpose(1, 2)
        u = F.silu(u)

        params = self.x_proj(u)
        delta_raw, b_param, c_param = torch.split(
            params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta_raw))
        y = self._selective_scan(u, delta, b_param, c_param)
        y = y * F.silu(gate)
        return self.out_proj(y)


class SceneMambaEncoder(nn.Module):
    """Bidirectional Mamba residual encoder for agent and lane scene tokens."""

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
        self.bidirectional = bidirectional
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(depth))
        self.layers = nn.ModuleList(
            SelectiveMambaSSM(dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(depth)
        )
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(depth))
        # sigmoid(-2.1972) = 0.1: begin close to baseline SHARP.
        self.residual_gates = nn.ParameterList(
            nn.Parameter(torch.tensor(-2.1972246)) for _ in range(depth)
        )

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


RUN_SCRIPT_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail

LAB_BASE="{base}"
EXPERIMENT_ROOT="{experiment_root}"
CODE_DIR="$EXPERIMENT_ROOT/Code"
RESULTS_DIR="{results_root}"
DATA_DIR="$LAB_BASE/Datasets/AV2/sharp_processed"
PYTHON_BIN="$LAB_BASE/Codes/envs/sharp/bin/python"
BATCH_SIZE="${{BATCH_SIZE:-8}}"
WORKERS="${{WORKERS:-6}}"

mkdir -p "$RESULTS_DIR"

source "$LAB_BASE/Codes/miniforge3/etc/profile.d/conda.sh"
conda activate "$LAB_BASE/Codes/envs/sharp"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$CODE_DIR:${{PYTHONPATH:-}}"
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

"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected 4 CUDA GPUs, found {{torch.cuda.device_count()}}")
print("CUDA GPUs:")
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

cd "$CODE_DIR"

# Real forward/backward smoke test before the 60-epoch run.
"$PYTHON_BIN" - <<'PY'
import torch
from src.model.layers.mamba_encoder import SceneMambaEncoder

device = torch.device("cuda:0")
module = SceneMambaEncoder(dim=128, depth=1).to(device)
x = torch.randn(4, 96, 128, device=device, requires_grad=True)
mask = torch.ones(4, 96, dtype=torch.bool, device=device)
mask[:, -8:] = False
y = module(x, mask)
loss = y.square().mean()
loss.backward()
grad_count = sum(
    parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
    for parameter in module.parameters()
)
param_count = sum(parameter.numel() for parameter in module.parameters())
print(f"MAMBA_SMOKE_TEST_OK shape={{tuple(y.shape)}} params={{param_count}} grads={{grad_count}}")
if y.shape != x.shape or grad_count == 0:
    raise SystemExit("Mamba smoke test failed")
PY

"$PYTHON_BIN" - <<'PY' > "$RESULTS_DIR/environment.txt"
import platform
import torch
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY

echo "Starting SHARP + Mamba on four GPUs"
echo "Per-GPU batch size: $BATCH_SIZE; global batch size: $((BATCH_SIZE * 4))"
echo "DataLoader workers per process: $WORKERS; total workers: $((WORKERS * 4))"

"$PYTHON_BIN" -u train.py \
  seed=2333 \
  gpus=4 \
  epochs=60 \
  batch_size="$BATCH_SIZE" \
  output_dir="$RESULTS_DIR/run" \
  datamodule.pl_module.data_root="$DATA_DIR" \
  datamodule.pl_module.num_workers="$WORKERS" \
  trainer.devices=4 \
  trainer.strategy=ddp_find_unused_parameters_false \
  trainer.sync_batchnorm=false \
  trainer.num_sanity_val_steps=0 \
  callbacks.0.save_top_k=3 \
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
    experiment_root = base / "Codes" / f"SHARP_AV2_MAMBA_ENCODER_V2_{stamp}"
    code_dir = experiment_root / "Code"
    results_root = base / "Results" / f"SHARP_AV2_MAMBA_ENCODER_V2_{stamp}"
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
        "    mamba_count = sum(parameter.numel() for _, parameter in mamba_params)\n"
        "    print(f'MAMBA_ACTIVE=True MAMBA_PARAMETER_TENSORS={len(mamba_params)} '"
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

    run_path = experiment_root / "run_av2_mamba_4gpu.sh"
    run_path.write_text(
        RUN_SCRIPT_TEMPLATE.format(
            base=base,
            experiment_root=experiment_root,
            results_root=results_root,
        ),
        encoding="utf-8",
    )
    make_executable(run_path)

    manifest = experiment_root / "EXPERIMENT.txt"
    manifest.write_text(
        "SHARP AV2 with native PyTorch selective Mamba scene encoder\n"
        f"Source code: {source_code}\n"
        f"Experiment code: {code_dir}\n"
        f"Results: {results_root}\n"
        "Placement: after positional token embedding, before dual training and "
        "instance-aware context streaming\n"
        "Mamba: depth=1, d_state=16, d_conv=4, expand=2, shared bidirectional scan\n",
        encoding="utf-8",
    )
    (base / "Codes/LATEST_SHARP_AV2_MAMBA_ENCODER_V2.txt").write_text(
        str(experiment_root) + "\n",
        encoding="utf-8",
    )

    print("SETUP_COMPLETE")
    print(f"EXPERIMENT_ROOT={experiment_root}")
    print(f"RESULTS_ROOT={results_root}")
    print(f"RUN_COMMAND={run_path}")


if __name__ == "__main__":
    main()
