#!/usr/bin/env python3
import argparse
import importlib.metadata
import sys
from pathlib import Path

import torch


VARIANTS = (
    "baseline",
    "mamba_agent_add",
    "mamba_future_replace",
)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()

    code = Path(args.code).resolve()
    sys.path.insert(0, str(code))

    from mamba_ssm.ops import selective_scan_interface
    from src.model.layers.mamba_layers import (
        MambaTrajectoryHead,
        ResidualMambaBlock,
    )
    from src.model.seam import Seam

    if getattr(selective_scan_interface, "selective_scan_cuda", None) is None:
        raise RuntimeError("The fused selective_scan CUDA extension is unavailable.")

    print("TORCH_VERSION=" + torch.__version__)
    print("TORCH_CUDA=" + str(torch.version.cuda))
    print("MAMBA_VERSION=" + importlib.metadata.version("mamba-ssm"))
    print("FUSED_SELECTIVE_SCAN_AVAILABLE=True")

    requested = [int(item) for item in args.gpus.split(",")]
    if torch.cuda.device_count() < len(requested):
        raise RuntimeError(
            f"Requested {len(requested)} GPUs but only {torch.cuda.device_count()} are visible."
        )

    for device_index in range(len(requested)):
        device = torch.device(f"cuda:{device_index}")

        history = ResidualMambaBlock(128).to(device)
        x = torch.randn(4, 30, 128, device=device, requires_grad=True)
        mask = torch.ones(4, 30, device=device)
        history(x, mask).float().square().mean().backward()

        future = MambaTrajectoryHead(128, 80, depth=2).to(device)
        query = torch.randn(2, 6, 128, device=device, requires_grad=True)
        output = future(query)
        if output.shape != (2, 6, 80, 2):
            raise RuntimeError(f"Unexpected future-head shape: {output.shape}")
        output.float().square().mean().backward()
        torch.cuda.synchronize(device)
        print(f"FUSED_MAMBA_CUDA_OK=cuda:{device_index}")

    shared = dict(
        embed_dim=128,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps=80,
        use_stream_encoder=True,
        use_stream_decoder=True,
        use_target_context=True,
        k=6,
        ma=False,
    )
    for variant in VARIANTS:
        model = Seam(variant=variant, **shared)
        names = tuple(name for name, _ in model.named_parameters())
        if variant == "baseline" and any("mamba" in name for name in names):
            raise RuntimeError("Baseline unexpectedly contains Mamba parameters.")
        if variant == "mamba_agent_add" and not any(
            name.startswith("agent_history_mamba") for name in names
        ):
            raise RuntimeError("Agent-history Mamba parameters were not registered.")
        if variant == "mamba_future_replace" and not any(
            name.startswith("decoder.loc.blocks") for name in names
        ):
            raise RuntimeError("Mamba trajectory-head parameters were not registered.")
        print(f"VARIANT_PREFLIGHT_OK={variant} parameters={count_parameters(model)}")

    print("ALL_VARIANTS_PREFLIGHT_OK=True")


if __name__ == "__main__":
    main()
