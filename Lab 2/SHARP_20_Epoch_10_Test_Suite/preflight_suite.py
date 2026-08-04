#!/usr/bin/env python3
"""Fail before expensive training if the ten-variant runtime is inconsistent."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from types import SimpleNamespace

import torch


VARIANTS = (
    "baseline",
    "confidence_gated_memory",
    "cross_window_consistency",
    "learned_temporal_pool",
    "uncertainty_target_context",
    "relative_geometry_bias",
    "kinematic_motion_stem",
    "endpoint_refinement_decoder",
    "lane_topology_graph",
    "agent_temporal_mamba",
)

MARKERS = {
    "confidence_gated_memory": "memory_update_gate",
    "learned_temporal_pool": "temporal_pool",
    "uncertainty_target_context": "uncertainty_target_context",
    "relative_geometry_bias": "relative_geometry_bias",
    "kinematic_motion_stem": "kinematic_stem",
    "endpoint_refinement_decoder": "endpoint_refinement",
    "lane_topology_graph": "lane_graph_refiner",
    "agent_temporal_mamba": "temporal_mamba",
}


def build_model():
    from src.model.sharp import Sharp

    return Sharp(
        embed_dim=128,
        future_steps=100,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        use_stream_encoder=True,
        use_stream_decoder=True,
        use_target_context=True,
        dual=True,
        biased_interaction=True,
        dm="av2",
        k=6,
    )


def optimizer_audit(model) -> None:
    from src.model.pl_modules import StreamLightningModule

    module = StreamLightningModule(
        model=model,
        num_grad_frame=10,
        optim=SimpleNamespace(
            lr=1.0e-4,
            weight_decay=1.0e-2,
            min_lr=1.0e-5,
            warmup_ratio=0.65,
            epochs=20,
        ),
    )
    optimizers, _ = module.configure_optimizers()
    optimized = {
        id(parameter)
        for group in optimizers[0].param_groups
        for parameter in group["params"]
    }
    trainable = {id(parameter) for parameter in module.parameters() if parameter.requires_grad}
    if optimized != trainable:
        raise RuntimeError(
            f"Optimizer mismatch: missing={len(trainable - optimized)}, "
            f"extra={len(optimized - trainable)}"
        )


def module_smoke_tests() -> None:
    from src.model.layers.ablation_modules import (
        EndpointRefinementHead,
        GeometricLaneGraphRefiner,
        KinematicMotionStem,
        LearnedTemporalPooling,
        MemoryUpdateGate,
        RelativeGeometryBias,
        UncertaintyTargetContext,
    )

    batch, length, dim = 4, 10, 128
    valid = torch.ones(batch, length, dtype=torch.bool)
    valid[:, :2] = False
    sequence = torch.randn(batch, length, dim, requires_grad=True)
    raw = torch.randn(batch, length, 5)
    outputs = [
        LearnedTemporalPooling(dim)(sequence, valid),
        KinematicMotionStem(dim)(raw, sequence, valid),
        MemoryUpdateGate(dim)(sequence, sequence * 0.5),
        RelativeGeometryBias(8)(torch.randn(2, 12, 2), torch.randn(2, 12)),
    ]
    radius, gate = UncertaintyTargetContext()(torch.randn(2, 6))
    outputs.extend([radius, gate])
    refined, logits = EndpointRefinementHead(dim, 100)(
        torch.randn(2, 6, dim), torch.randn(2, 6, 100, 4), torch.randn(2, 6)
    )
    outputs.extend([refined, logits])
    outputs.append(
        GeometricLaneGraphRefiner(dim)(
            torch.randn(2, 16, dim),
            torch.randn(2, 16, 2),
            torch.randn(2, 16),
            torch.ones(2, 16, dtype=torch.bool),
        )
    )
    if not all(torch.isfinite(output).all() for output in outputs):
        raise RuntimeError("A non-Mamba ablation smoke test produced non-finite output")


def mamba_smoke_test() -> None:
    from src.model.layers.temporal_mamba_replacement import (
        TemporalMambaStack,
        stable_cuda_available,
    )

    if not stable_cuda_available():
        raise RuntimeError("Mamba CUDA selective scan is unavailable")
    module = TemporalMambaStack().cuda()
    x = torch.randn(32, 10, 128, device="cuda", requires_grad=True)
    valid = torch.ones(32, 10, device="cuda", dtype=torch.bool)
    valid[:, :2] = False
    output = module(x, valid)
    output.square().mean().backward()
    if not torch.isfinite(output).all() or x.grad is None or not torch.isfinite(x.grad).all():
        raise RuntimeError("Mamba CUDA smoke test failed")
    print(f"MAMBA_CUDA_SMOKE_OK shape={tuple(output.shape)}")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise SystemExit(f"Expected four CUDA GPUs, found {torch.cuda.device_count()}")
    print("CUDA GPUs:")
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index))

    config = Path("conf/model/Sharp_av2.yaml").read_text(encoding="utf-8")
    required = ("lr: 1e-4", "min_lr: 1e-5", "weight_decay: 1e-2", "warmup_ratio: 0.65")
    for value in required:
        if value not in config:
            raise RuntimeError(f"Missing controlled config value: {value}")

    module_smoke_tests()
    for variant in VARIANTS:
        os.environ["SHARP_EXPERIMENT_VARIANT"] = variant
        model = build_model()
        names = [name for name, _ in model.named_parameters()]
        marker = MARKERS.get(variant)
        if marker and not any(marker in name for name in names):
            raise RuntimeError(f"{variant}: expected marker {marker} is absent")
        if variant == "agent_temporal_mamba":
            if any("h_embed" in name for name in names):
                raise RuntimeError("Mamba variant still contains temporal MHA parameters")
        elif any("temporal_mamba" in name for name in names):
            raise RuntimeError(f"{variant}: unexpected Mamba parameters")
        optimizer_audit(model)
        print(
            f"VARIANT_PREFLIGHT_OK={variant} "
            f"parameters={sum(p.numel() for p in model.parameters())}"
        )
        del model
        gc.collect()

    os.environ["SHARP_EXPERIMENT_VARIANT"] = "agent_temporal_mamba"
    mamba_smoke_test()
    print("ALL_TEN_VARIANTS_PREFLIGHT_OK=True")


if __name__ == "__main__":
    main()
