#!/usr/bin/env python3
"""Apply compatibility fixes to the existing Lab 3 attention experiment."""

from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path

BASE = Path("/home/server01/M")
VARIANTS = ("qknorm", "talking_heads", "qknorm_talking_heads")
ENV_EXPORTS = """\
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export NO_COLOR=1
export RICH_NO_COLOR=1
"""
SITECUSTOMIZE = """\
import warnings

import numpy as np

for _name, _value in (("bool", bool), ("int", int), ("float", float)):
    if _name not in np.__dict__:
        setattr(np, _name, _value)

# Compatibility warnings from pinned third-party packages. These filters do not
# suppress training, CUDA, NCCL, numerical, or exception warnings.
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
"""


def pointer(path: Path) -> Path:
    value = path.read_text().strip()
    if not value:
        raise RuntimeError(f"Empty pointer file: {path}")
    return Path(value)


def patch_timm_import(path: Path) -> None:
    text = path.read_text()
    updated = text.replace(
        "from timm.models.layers import DropPath",
        "from timm.layers import DropPath",
    )
    if updated != text:
        path.write_text(updated)


def patch_runner(path: Path) -> None:
    text = path.read_text()
    if "export PYTHONUTF8=1" not in text:
        marker = 'export PYTHONPATH="$PATCH:$CODE:${PYTHONPATH:-}"\n'
        if marker not in text:
            raise RuntimeError(f"PYTHONPATH marker not found in {path}")
        text = text.replace(marker, marker + ENV_EXPORTS, 1)
        path.write_text(text)


def patch_validation_batch_size(path: Path) -> None:
    """Make Lightning validation aggregation explicit without changing training."""
    text = path.read_text()
    old = '''        self.log_dict(
            reg_loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
'''
    new = '''        self.log_dict(
            reg_loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(data[-1]["scenario_id"]),
        )
'''
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(
            f"Expected one streaming validation log block in {path}, "
            f"found {text.count(old)}"
        )
    path.write_text(text.replace(old, new, 1))


def smoke_test(path: Path, variant: str) -> None:
    module_name = f"_sharp_attention_smoke_{variant}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load attention module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    import torch

    torch.manual_seed(2333)
    layer = module.VariantMultiheadAttention(
        embed_dim=128,
        num_heads=8,
        dropout=0.0,
        batch_first=True,
        kdim=64,
        vdim=96,
    )
    query = torch.randn(2, 7, 128, requires_grad=True)
    key = torch.randn(2, 9, 64, requires_grad=True)
    value = torch.randn(2, 9, 96, requires_grad=True)
    attn_mask = torch.zeros(7, 9, dtype=torch.bool)
    attn_mask[:, -1] = True
    padding_mask = torch.zeros(2, 9, dtype=torch.bool)
    padding_mask[1, -2:] = True

    output, weights = layer(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=padding_mask,
        need_weights=True,
    )
    if output.shape != (2, 7, 128) or weights is None:
        raise RuntimeError(f"{variant}: unexpected output shape")
    if not torch.isfinite(output).all() or not torch.isfinite(weights).all():
        raise RuntimeError(f"{variant}: non-finite smoke-test output")
    output.square().mean().backward()
    if any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    ):
        raise RuntimeError(f"{variant}: invalid smoke-test gradients")


def main() -> int:
    root = pointer(BASE / "Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")
    results = pointer(BASE / "Results/LATEST_SHARP_ATTENTION_ABLATION.txt")
    runner = root / "run_variant.sh"
    runtime_patch = root / "runtime_patch/sitecustomize.py"

    if not runner.is_file():
        raise FileNotFoundError(runner)
    runtime_patch.parent.mkdir(parents=True, exist_ok=True)
    runtime_patch.write_text(SITECUSTOMIZE)
    patch_runner(runner)

    for variant in VARIANTS:
        layers = root / "variants" / variant / "Code/src/model/layers"
        attention = layers / "attention_variants.py"
        if not attention.is_file():
            raise FileNotFoundError(attention)
        for name in ("custom_transformer_blocks.py", "transformer_blocks.py"):
            path = layers / name
            patch_timm_import(path)
            py_compile.compile(str(path), doraise=True)
        pl_modules = root / "variants" / variant / "Code/src/model/pl_modules.py"
        patch_validation_batch_size(pl_modules)
        py_compile.compile(str(pl_modules), doraise=True)
        py_compile.compile(str(attention), doraise=True)
        smoke_test(attention, variant)
        print(f"ATTENTION_SMOKE_TEST_OK={variant}")
        print(f"VALIDATION_BATCH_SIZE_LOGGING_OK={variant}")

    print(f"RUNTIME_COMPATIBILITY_PATCHED={root}")
    print(f"RESULTS_ROOT={results}")
    print("TRAINING_CONFIGURATION_CHANGED=False")
    print("VALIDATION_LOG_BATCH_SIZE_EXPLICIT=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())