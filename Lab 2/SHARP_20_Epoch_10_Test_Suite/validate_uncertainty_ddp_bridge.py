#!/usr/bin/env python3
"""Audit the zero-valued autograd bridge used by uncertainty context."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


MARKER = "uncertainty_ddp_zero = target_pos.new_zeros(())"
LOSS_MARKER = "loss = loss + parameter.sum() * 0.0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()

    experiment = args.experiment_root.resolve()
    code = experiment / "Code"
    sharp_path = code / "src/model/sharp.py"
    lightning_path = code / "src/model/pl_modules.py"
    source = sharp_path.read_text(encoding="utf-8")
    if source.count(MARKER) != 1:
        raise SystemExit("FATAL: patched SHARP source marker is missing or duplicated")
    compile(source, str(sharp_path), "exec")
    lightning_source = lightning_path.read_text(encoding="utf-8")
    if lightning_source.count(LOSS_MARKER) != 1:
        raise SystemExit("FATAL: uncertainty loss bridge is missing or duplicated")
    compile(lightning_source, str(lightning_path), "exec")

    sys.path.insert(0, str(code))
    import torch
    from src.model.layers.ablation_modules import UncertaintyTargetContext

    module = UncertaintyTargetContext()
    original_loss = torch.tensor(3.25, requires_grad=True)
    loss = original_loss
    for parameter in module.parameters():
        loss = loss + parameter.sum() * 0.0

    if loss.detach().item() != original_loss.detach().item():
        raise SystemExit("FATAL: loss bridge changed the loss value")
    loss.backward()

    missing = []
    nonzero = []
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif torch.count_nonzero(parameter.grad).item() != 0:
            nonzero.append(name)
    if missing:
        raise SystemExit(f"FATAL: bridge did not connect parameters: {missing}")
    if nonzero:
        raise SystemExit(f"FATAL: bridge generated nonzero gradients: {nonzero}")

    print("UNCERTAINTY_DDP_BRIDGE_AUDIT_OK=True")
    print(f"CONNECTED_PARAMETER_TENSORS={sum(1 for _ in module.parameters())}")
    print("LOSS_VALUE_UNCHANGED=True")
    print("BRIDGE_GRADIENTS_ARE_ZERO=True")


if __name__ == "__main__":
    main()
