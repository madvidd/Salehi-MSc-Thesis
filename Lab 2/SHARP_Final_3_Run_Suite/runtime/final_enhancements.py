"""Composed accuracy-oriented modules for the final SHARP experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RelativeGeometryBias(nn.Module):
    """Learn a zero-initialized per-head bias from relative pose and heading."""

    def __init__(self, num_heads: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, centers: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        relative = (centers.unsqueeze(2) - centers.unsqueeze(1)) / 30.0
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        delta_angle = angles.unsqueeze(2) - angles.unsqueeze(1)
        geometry = torch.cat(
            (
                relative,
                distance,
                torch.cos(delta_angle).unsqueeze(-1),
                torch.sin(delta_angle).unsqueeze(-1),
            ),
            dim=-1,
        )
        bias = self.net(geometry).permute(0, 3, 1, 2).contiguous()
        return bias.view(-1, centers.size(1), centers.size(1))


class UncertaintyTargetContext(nn.Module):
    """Adapt target-context radius and feature strength to prior mode entropy."""

    def __init__(self) -> None:
        super().__init__()
        self.min_radius = nn.Parameter(torch.tensor(24.0))
        self.radius_span = nn.Parameter(torch.tensor(16.0))
        self.gate = nn.Linear(2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(
            probability * torch.log(probability.clamp_min(1.0e-8)), dim=-1
        ) / math.log(probability.size(-1))
        radius = self.min_radius.clamp(15.0, 35.0) + self.radius_span.clamp(
            1.0, 25.0
        ) * (1.0 - probability)
        gate_input = torch.stack(
            (probability, entropy.unsqueeze(-1).expand_as(probability)), dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_input)).squeeze(-1)
        return radius, gate
