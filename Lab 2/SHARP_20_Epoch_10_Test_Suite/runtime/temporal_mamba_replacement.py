"""Unidirectional Mamba replacement for SHARP's temporal agent attention stack."""

from __future__ import annotations

import torch
import torch.nn as nn
import mamba_ssm.modules.mamba_simple as mamba_simple
from mamba_ssm.modules.mamba_simple import Mamba
from timm.layers import DropPath


def configure_stable_cuda_backend() -> None:
    """Use CUDA selective scan while avoiding the optional causal-conv extension."""
    mamba_simple.causal_conv1d_fn = None
    mamba_simple.causal_conv1d_update = None


def stable_cuda_available() -> bool:
    return (
        mamba_simple.causal_conv1d_fn is None
        and mamba_simple.selective_scan_fn is not None
    )


class ResidualMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid = valid_mask.unsqueeze(-1).to(x.dtype)
        update = self.mamba((self.norm(x) * valid).contiguous())
        return (x + self.drop_path(update)) * valid


class TemporalMambaStack(nn.Module):
    """Four chronological Mamba blocks replacing four temporal MHA blocks."""

    def __init__(
        self,
        dim: int = 128,
        depth: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        drop_path: float = 0.2,
        actor_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        configure_stable_cuda_backend()
        if not stable_cuda_available():
            raise RuntimeError("Fused CUDA selective scan is unavailable")
        rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            ResidualMambaBlock(dim, d_state, d_conv, expand, rates[index])
            for index in range(depth)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.actor_chunk_size = actor_chunk_size

    def _forward_chunk(
        self, x: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, valid_mask)
        return self.final_norm(x) * valid_mask.unsqueeze(-1).to(x.dtype)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, x.size(0), self.actor_chunk_size):
            stop = start + self.actor_chunk_size
            outputs.append(self._forward_chunk(x[start:stop], valid_mask[start:stop]))
        return torch.cat(outputs, dim=0)
