#!/usr/bin/env python3
"""Stress the Lab 3 attention kernels on all three GPUs before long runs."""

from __future__ import annotations

import faulthandler
import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

BASE = Path("/home/server01/M")
VARIANTS = ("qknorm", "talking_heads", "qknorm_talking_heads")
ITERATIONS = int(os.environ.get("LAB3_CUDA_STRESS_ITERATIONS", "300"))


def pointer(path: Path) -> Path:
    value = path.read_text().strip()
    if not value:
        raise RuntimeError(f"Empty pointer file: {path}")
    return Path(value)


def load_attention(path: Path, variant: str):
    module_name = f"_lab3_cuda_stress_{variant}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load attention module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def check_runtime() -> None:
    if not torch.__version__.startswith("2.8.0"):
        raise RuntimeError(f"Expected PyTorch 2.8.0, found {torch.__version__}")
    if torch.version.cuda != "12.6":
        raise RuntimeError(f"Expected CUDA runtime 12.6, found {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 3:
        raise RuntimeError(
            f"Expected at least three CUDA devices, found {torch.cuda.device_count()}"
        )


def stress_variant(root: Path, variant: str, device: torch.device) -> None:
    path = (
        root
        / "variants"
        / variant
        / "Code/src/model/layers/attention_variants.py"
    )
    module = load_attention(path, variant)
    layer = module.VariantMultiheadAttention(
        embed_dim=128,
        num_heads=8,
        dropout=0.0,
        batch_first=True,
        kdim=64,
        vdim=96,
    ).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)

    for step in range(ITERATIONS):
        generator = torch.Generator(device=device)
        generator.manual_seed(2333 + step)
        query = torch.randn(
            8, 48, 128, generator=generator, device=device, requires_grad=True
        )
        key = torch.randn(
            8, 56, 64, generator=generator, device=device, requires_grad=True
        )
        value = torch.randn(
            8, 56, 96, generator=generator, device=device, requires_grad=True
        )
        attn_mask = torch.zeros(48, 56, dtype=torch.bool, device=device)
        attn_mask[:, -1] = True
        padding_mask = torch.zeros(8, 56, dtype=torch.bool, device=device)
        padding_mask[-1, -4:] = True

        optimizer.zero_grad(set_to_none=True)
        output, weights = layer(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=True,
        )
        if weights is None:
            raise RuntimeError(f"{variant}: attention weights were not returned")
        loss = output.float().square().mean() + weights.float().square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{variant}: non-finite loss at step {step}")
        loss.backward()
        for name, parameter in layer.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"{variant}: invalid gradient for {name}")
        optimizer.step()
        if step % 25 == 0:
            collective = loss.detach().clone()
            dist.all_reduce(collective)
            if not torch.isfinite(collective):
                raise RuntimeError(f"{variant}: non-finite NCCL collective")

    torch.cuda.synchronize(device)
    del optimizer, layer
    torch.cuda.empty_cache()


def main() -> int:
    faulthandler.enable(all_threads=True)
    check_runtime()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    root = pointer(BASE / "Codes/LATEST_SHARP_ATTENTION_ABLATION.txt")

    try:
        for variant in VARIANTS:
            stress_variant(root, variant, device)
            dist.barrier(device_ids=[local_rank])
            if dist.get_rank() == 0:
                print(
                    f"CUDA_DDP_ATTENTION_STRESS_OK={variant} "
                    f"iterations={ITERATIONS} ranks={dist.get_world_size()}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()

    if local_rank == 0:
        print(f"LAB3_CUDA_RUNTIME_VALIDATED={torch.__version__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
