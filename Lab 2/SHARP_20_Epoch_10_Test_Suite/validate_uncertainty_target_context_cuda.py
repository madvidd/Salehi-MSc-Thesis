#!/usr/bin/env python3
"""Validate equivalence and gradients for the checked target-context remap."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import torch


MARKER = "target_destination = torch.nonzero("
OLD_OPERATION = "container[target_valid.view(-1)] ="


def original_remap(
    target_valid: torch.Tensor,
    compressed: torch.Tensor,
    compressed_mask: torch.Tensor,
) -> torch.Tensor:
    rows, original_width = target_valid.shape
    channels = compressed.size(-1)
    output = compressed.new_zeros(rows * original_width, channels)
    output[target_valid.reshape(-1)] = compressed.reshape(-1, channels)[
        ~compressed_mask.reshape(-1)
    ]
    return output


def checked_remap(
    target_valid: torch.Tensor,
    compressed: torch.Tensor,
    compressed_mask: torch.Tensor,
) -> torch.Tensor:
    rows, original_width = target_valid.shape
    channels = compressed.size(-1)
    output = compressed.new_zeros(rows * original_width, channels)
    destination = torch.nonzero(
        target_valid.reshape(-1), as_tuple=False
    ).squeeze(-1)
    source = torch.nonzero(
        ~compressed_mask.reshape(-1), as_tuple=False
    ).squeeze(-1)
    if destination.numel() != source.numel():
        raise RuntimeError("Synthetic remap count mismatch")
    values = compressed.reshape(-1, channels).index_select(0, source)
    return torch.index_copy(output, 0, destination, values)


def run_equivalence_trials() -> None:
    generator = torch.Generator().manual_seed(2333)
    for trial in range(128):
        rows = 1 + trial % 19
        width = 3 + trial % 23
        channels = 2 + trial % 13
        target_valid = torch.rand(rows, width, generator=generator) > 0.45
        target_valid[:, 0] = True
        counts = target_valid.sum(dim=1)
        padded_width = int(counts.max().item())
        compressed_mask = (
            torch.arange(padded_width).unsqueeze(0) >= counts.unsqueeze(1)
        )

        first = torch.randn(
            rows,
            padded_width,
            channels,
            generator=generator,
            requires_grad=True,
        )
        second = first.detach().clone().requires_grad_(True)

        original = original_remap(target_valid, first, compressed_mask)
        checked = checked_remap(target_valid, second, compressed_mask)
        torch.testing.assert_close(original, checked, rtol=0, atol=0)

        weights = torch.randn(original.shape, generator=generator)
        (original * weights).sum().backward()
        (checked * weights).sum().backward()
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    args = parser.parse_args()

    model_path = args.experiment_root.resolve() / "Code/src/model/sharp.py"
    source = model_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(model_path))

    if source.count(MARKER) != 1:
        raise SystemExit("FATAL: checked target remap marker is missing or duplicated")
    if OLD_OPERATION in source:
        raise SystemExit("FATAL: unsafe Boolean target write remains in the model")

    run_equivalence_trials()
    print("TARGET_REMAP_STATIC_AUDIT=True")
    print("TARGET_REMAP_FORWARD_EQUIVALENT=True")
    print("TARGET_REMAP_GRADIENT_EQUIVALENT=True")


if __name__ == "__main__":
    main()
