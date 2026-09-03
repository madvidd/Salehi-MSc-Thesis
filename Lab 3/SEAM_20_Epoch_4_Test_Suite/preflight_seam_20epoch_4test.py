#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


VARIANTS = (
    "baseline",
    "uncertainty_target_context",
    "relative_geometry_bias",
    "qknorm",
)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def optimizer_audit(lightning_module):
    optimizers, schedulers = lightning_module.configure_optimizers()
    if len(optimizers) != 1 or len(schedulers) != 1:
        raise RuntimeError("Unexpected optimiser/scheduler count")
    model_parameters = {
        id(parameter) for parameter in lightning_module.parameters() if parameter.requires_grad
    }
    optimizer_parameters = {
        id(parameter)
        for group in optimizers[0].param_groups
        for parameter in group["params"]
    }
    if model_parameters != optimizer_parameters:
        raise RuntimeError("Optimiser parameter coverage is incomplete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True)
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()

    code = Path(args.code).resolve()
    sys.path.insert(0, str(code))

    from src.model.layers.controlled_ablation import (
        QKNormMultiheadAttention,
        RelativeGeometryBias,
        UncertaintyTargetContext,
    )
    from src.model.pl_modules import StreamLightningModule
    from src.model.seam import Seam

    print("TORCH_VERSION=" + torch.__version__)
    print("TORCH_CUDA=" + str(torch.version.cuda))
    print("CUDA_AVAILABLE=" + str(torch.cuda.is_available()))

    requested = [item for item in args.gpus.split(",") if item.strip()]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() < len(requested):
        raise RuntimeError(
            f"Requested {len(requested)} GPUs but only {torch.cuda.device_count()} are visible"
        )
    for index in range(len(requested)):
        properties = torch.cuda.get_device_properties(index)
        if properties.major != 7 or properties.minor != 5:
            raise RuntimeError(
                f"cuda:{index} is compute capability {properties.major}.{properties.minor}; "
                "this Lab 3 package expects RTX 2080 Ti capability 7.5"
            )
        print(
            f"CUDA_DEVICE_OK=cuda:{index};name={properties.name};"
            f"memory={properties.total_memory}"
        )

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
    counts = {}
    for variant in VARIANTS:
        model = Seam(variant=variant, **shared)
        counts[variant] = count_parameters(model)
        modules = tuple(model.modules())

        has_qknorm = any(isinstance(module, QKNormMultiheadAttention) for module in modules)
        has_geometry = any(isinstance(module, RelativeGeometryBias) for module in modules)
        has_uncertainty = any(
            isinstance(module, UncertaintyTargetContext) for module in modules
        )
        expected = {
            "baseline": (False, False, False),
            "uncertainty_target_context": (False, False, True),
            "relative_geometry_bias": (False, True, False),
            "qknorm": (True, False, False),
        }[variant]
        observed = (has_qknorm, has_geometry, has_uncertainty)
        if observed != expected:
            raise RuntimeError(
                f"Variant isolation failed for {variant}: expected {expected}, got {observed}"
            )

        optim = SimpleNamespace(
            lr=0.001,
            weight_decay=0.01,
            min_lr=0.00001,
            warmup_ratio=0.167,
            epochs=20,
        )
        lightning_module = StreamLightningModule(
            model=model, optim=optim, ma=False, num_grad_frame=3
        )
        optimizer_audit(lightning_module)
        print(f"VARIANT_PREFLIGHT_OK={variant};parameters={counts[variant]}")

    if counts["baseline"] != 4_604_769:
        raise RuntimeError(
            "Baseline parameter count differs from the verified previous SEAM "
            f"baseline: {counts['baseline']} != 4604769"
        )
    expected_counts = {
        "baseline": 4_604_769,
        "uncertainty_target_context": 4_604_777,
        "relative_geometry_bias": 4_605_225,
        "qknorm": 4_604_929,
    }
    if counts != expected_counts:
        raise RuntimeError(
            f"Variant parameter counts do not match the audited design: {counts}"
        )

    source = (code / "src/model/pl_modules.py").read_text(encoding="utf-8")
    if "batch_size=batch_size" not in source:
        raise RuntimeError("Validation logging does not set batch_size explicitly")
    if "timm.models.layers" in "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (code / "src/model/layers").glob("*.py")
    ):
        raise RuntimeError("Deprecated timm.models.layers import remains")

    print("OPTIMIZER_COVERAGE_OK=True")
    print("BASELINE_PARAMETER_PARITY_OK=True")
    print("VARIANT_ISOLATION_OK=True")
    print("WARNING_HARDENING_OK=True")
    print("ALL_VARIANTS_PREFLIGHT_OK=True")


if __name__ == "__main__":
    main()
