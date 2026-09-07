"""Isolated SHARP ablation modules used by the 20-epoch comparison suite."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LearnedTemporalPooling(nn.Module):
    """Learn one normalized weight per valid observation instead of max pooling."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1) * valid_mask.to(x.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class KinematicMotionStem(nn.Module):
    """Add masked first- and second-order motion differences before encoding."""

    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.layer_scale = nn.Parameter(torch.full((dim,), 0.01))

    @staticmethod
    def _difference(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], dim=1)

    def forward(
        self,
        raw_features: torch.Tensor,
        projected: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        position = raw_features[..., :2]
        speed = raw_features[..., 2:3]
        velocity = self._difference(position)
        acceleration = self._difference(velocity)
        speed_delta = self._difference(speed)
        kinematics = torch.cat([velocity, acceleration, speed_delta], dim=-1)
        kinematics = kinematics * valid_mask.unsqueeze(-1).to(kinematics.dtype)
        update = self.net(kinematics) * self.layer_scale
        return projected + update * valid_mask.unsqueeze(-1).to(update.dtype)


class MemoryUpdateGate(nn.Module):
    """Reliability gate between the current token and streamed memory update."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim * 3, dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, current: torch.Tensor, updated: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(
            self.gate(torch.cat([current, updated, updated - current], dim=-1))
        )
        return current + gate * (updated - current)


class RelativeGeometryBias(nn.Module):
    """Per-head additive attention bias from relative pose and heading."""

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
            [
                relative,
                distance,
                torch.cos(delta_angle).unsqueeze(-1),
                torch.sin(delta_angle).unsqueeze(-1),
            ],
            dim=-1,
        )
        bias = self.net(geometry).permute(0, 3, 1, 2).contiguous()
        return bias.view(-1, centers.size(1), centers.size(1))


class UncertaintyTargetContext(nn.Module):
    """Adapt endpoint RoI radius and feature strength to previous mode uncertainty."""

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
            [probability, entropy.unsqueeze(-1).expand_as(probability)], dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_input)).squeeze(-1)
        return radius, gate


class EndpointRefinementHead(nn.Module):
    """Refine DETR mode queries using their coarse predicted endpoints."""

    def __init__(self, dim: int, future_steps: int) -> None:
        super().__init__()
        self.future_steps = future_steps
        self.endpoint_embed = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.trajectory_delta = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, future_steps * 2),
        )
        self.score_delta = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        self.trajectory_scale = nn.Parameter(torch.tensor(0.01))
        self.score_scale = nn.Parameter(torch.tensor(0.01))

    def forward(
        self,
        mode_features: torch.Tensor,
        coarse_trajectory: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        endpoint = coarse_trajectory[..., -1, :2]
        query = self.norm(mode_features + self.endpoint_embed(endpoint))
        delta = self.trajectory_delta(query).view(
            query.size(0), query.size(1), self.future_steps, 2
        )
        refined_xy = coarse_trajectory[..., :2] + self.trajectory_scale * delta
        refined = torch.cat([refined_xy, coarse_trajectory[..., 2:]], dim=-1)
        logits = coarse_logits + self.score_scale * self.score_delta(query).squeeze(-1)
        return refined, logits


class GeometricLaneGraphRefiner(nn.Module):
    """Geometry-derived sparse lane graph message passing without new map labels."""

    def __init__(self, dim: int, neighbors: int = 8, radius: float = 40.0) -> None:
        super().__init__()
        self.neighbors = neighbors
        self.radius = radius
        self.edge_score = nn.Sequential(
            nn.Linear(4, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.layer_scale = nn.Parameter(torch.full((dim,), 0.01))

    def forward(
        self,
        lane_features: torch.Tensor,
        centers: torch.Tensor,
        angles: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, lanes, dim = lane_features.shape
        if lanes == 0:
            return lane_features
        neighbors = min(self.neighbors, lanes)
        distance = torch.cdist(centers.float(), centers.float())
        valid_pair = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        distance = distance.masked_fill(~valid_pair, float("inf"))
        distance = distance.masked_fill(
            torch.eye(lanes, device=distance.device, dtype=torch.bool).unsqueeze(0),
            float("inf"),
        )
        neighbor_distance, neighbor_index = torch.topk(
            distance, k=neighbors, dim=-1, largest=False
        )
        offsets = torch.arange(batch, device=lane_features.device).view(-1, 1, 1) * lanes
        flat_index = neighbor_index + offsets
        neighbor_features = lane_features.reshape(batch * lanes, dim)[flat_index]
        neighbor_centers = centers.reshape(batch * lanes, 2)[flat_index]
        neighbor_angles = angles.reshape(batch * lanes)[flat_index]
        relative = (neighbor_centers - centers.unsqueeze(2)) / self.radius
        delta_angle = neighbor_angles - angles.unsqueeze(2)
        edge_geometry = torch.cat(
            [
                relative,
                torch.cos(delta_angle).unsqueeze(-1),
                torch.sin(delta_angle).unsqueeze(-1),
            ],
            dim=-1,
        )
        edge_valid = torch.isfinite(neighbor_distance) & (
            neighbor_distance <= self.radius
        )
        score = self.edge_score(edge_geometry).squeeze(-1).masked_fill(
            ~edge_valid, -1.0e4
        )
        weight = torch.softmax(score, dim=-1) * edge_valid.to(score.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        message = torch.sum(neighbor_features * weight.unsqueeze(-1), dim=2)
        update = self.update(torch.cat([lane_features, message], dim=-1))
        refined = lane_features + self.layer_scale * update
        return torch.where(valid_mask.unsqueeze(-1), refined, lane_features)


def endpoint_auxiliary_loss(
    coarse_trajectory: torch.Tensor,
    coarse_logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Winner-takes-all auxiliary loss on the coarse decoder output."""
    horizon = target.size(-2)
    coarse = coarse_trajectory[:, :, :horizon, :2]
    displacement = torch.linalg.vector_norm(
        coarse - target.unsqueeze(1), dim=-1
    ).sum(dim=-1)
    best_mode = torch.argmin(displacement, dim=-1)
    best = coarse[torch.arange(coarse.size(0), device=coarse.device), best_mode]
    return F.smooth_l1_loss(best, target) + F.cross_entropy(
        coarse_logits, best_mode.detach()
    )
