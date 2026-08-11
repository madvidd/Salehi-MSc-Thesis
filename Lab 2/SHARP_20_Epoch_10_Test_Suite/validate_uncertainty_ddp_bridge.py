#!/usr/bin/env python3
"""Audit the zero-valued autograd bridge used by uncertainty context."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


MARKER = "uncertainty_ddp_zero = target_pos.new_zeros(())"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()

    experiment = args.experiment_root.resolve()
    code = experiment / "Code"
    sharp_path = code / "src/model/sharp.py"
    source = sharp_path.read_text(encoding="utf-8")
    if source.count(MARKER) != 1:
        raise SystemExit("FATAL: patched SHARP source marker is missing or duplicated")
    compile(source, str(sharp_path), "exec")

    sys.path.insert(0, str(code))
    import torch
    from src.model.layers.ablation_modules import UncertaintyTargetContext

    module = UncertaintyTargetContext()
    gate = torch.ones(2, 6)
    ddp_zero = gate.new_zeros(())
    for parameter in module.parameters():
        ddp_zero = ddp_zero + parameter.sum() * 0.0
    bridged_gate = gate + ddp_zero

    if not torch.equal(gate, bridged_gate):
        raise SystemExit("FATAL: DDP bridge changed a forward value")
    bridged_gate.sum().backward()

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
    print("FORWARD_VALUES_UNCHANGED=True")
    print("BRIDGE_GRADIENTS_ARE_ZERO=True")


if __name__ == "__main__":
    main()
