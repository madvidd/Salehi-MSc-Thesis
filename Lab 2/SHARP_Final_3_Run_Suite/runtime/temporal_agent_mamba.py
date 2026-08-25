"""Small residual Mamba branch over chronological SHARP agent histories."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import selective_scan_cuda
import mamba_ssm.modules.mamba_simple as mamba_simple
from mamba_ssm.modules.mamba_simple import Mamba


def configure_stable_cuda_backend() -> None:
    """Use cuDNN Conv1d while retaining the fused selective-scan CUDA kernel."""
    mamba_simple.causal_conv1d_fn = None
    mamba_simple.causal_conv1d_update = None


configure_stable_cuda_backend()


def stable_cuda_available() -> bool:
    return (
        mamba_simple.causal_conv1d_fn is None
        and mamba_simple.selective_scan_fn is not None
        and hasattr(selective_scan_cuda, "fwd")
    )


def backend_summary() -> str:
    return (
        "CAUSAL_CONV_BACKEND=torch_cuda_cudnn_conv1d "
        "SELECTIVE_SCAN_BACKEND=fused_selective_scan_cuda"
    )


class TemporalAgentMamba(nn.Module):
    """Gated bidirectional residual Mamba over valid observations per agent."""

    def __init__(
        self,
        dim: int,
        d_state: int = 8,
        d_conv: int = 3,
        expand: int = 1,
        dropout: float = 0.1,
        layer_scale_init: float = 0.01,
        agent_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if not stable_cuda_available():
            raise RuntimeError("Stable fused-CUDA Mamba backend is unavailable")
        if agent_chunk_size < 1:
            raise ValueError("agent_chunk_size must be positive")

        self.norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.backward_mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        self.direction_logits = nn.Parameter(torch.zeros(dim))
        self.layer_scale = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.dropout = nn.Dropout(dropout)
        self.agent_chunk_size = agent_chunk_size

    @staticmethod
    def _compact_valid(
        value: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = value.shape
        compact_rank = valid_mask.long().cumsum(dim=1) - 1
        row = torch.arange(batch, device=value.device).unsqueeze(1).expand(
            batch, length
        )
        compact = torch.zeros_like(value)
        compact[row[valid_mask], compact_rank[valid_mask]] = value[valid_mask]
        lengths = valid_mask.sum(dim=1)
        compact_valid = (
            torch.arange(length, device=value.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        return compact, compact_rank, compact_valid

    @staticmethod
    def _reverse_valid_prefix(
        value: torch.Tensor,
        lengths: torch.Tensor,
        valid_prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, length, dim = value.shape
        positions = torch.arange(length, device=value.device).unsqueeze(0)
        reverse_index = (lengths.unsqueeze(1) - 1 - positions).clamp(
            0, length - 1
        )
        gather_index = reverse_index.unsqueeze(-1).expand(-1, -1, dim)
        reversed_value = value.gather(1, gather_index)
        reversed_value = reversed_value * valid_prefix.unsqueeze(-1).to(value.dtype)
        return reversed_value, gather_index

    def _run_fixed_chunks(self, module: Mamba, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] == 0:
            return value
        outputs = []
        for start in range(0, value.shape[0], self.agent_chunk_size):
            chunk = value[start : start + self.agent_chunk_size].contiguous()
            real_rows = chunk.shape[0]
            if real_rows < self.agent_chunk_size:
                chunk = F.pad(
                    chunk,
                    (0, 0, 0, 0, 0, self.agent_chunk_size - real_rows),
                )
            outputs.append(module(chunk.contiguous())[:real_rows])
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        value: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim != 3 or valid_mask.shape != value.shape[:2]:
            raise ValueError(
                "Expected [agents,time,dim] and matching mask, got "
                f"{tuple(value.shape)} and {tuple(valid_mask.shape)}"
            )
        normalized = self.norm(value)
        compact, compact_rank, compact_valid = self._compact_valid(
            normalized, valid_mask
        )
        compact = compact.contiguous()
        lengths = compact_valid.sum(dim=1)
        forward_out = self._run_fixed_chunks(self.forward_mamba, compact)
        backward_input, reverse_index = self._reverse_valid_prefix(
            compact, lengths, compact_valid
        )
        backward_reversed = self._run_fixed_chunks(
            self.backward_mamba, backward_input.contiguous()
        )
        backward_out = backward_reversed.gather(1, reverse_index)
        direction_gate = torch.sigmoid(self.direction_logits).view(1, 1, -1)
        compact_update = (
            direction_gate * forward_out
            + (1.0 - direction_gate) * backward_out
        )
        compact_update = compact_update * compact_valid.unsqueeze(-1).to(
            value.dtype
        )

        batch, length, _ = value.shape
        row = torch.arange(batch, device=value.device).unsqueeze(1).expand(
            batch, length
        )
        update = torch.zeros_like(value)
        update[valid_mask] = compact_update[
            row[valid_mask], compact_rank[valid_mask]
        ]
        return value + self.layer_scale.view(1, 1, -1) * self.dropout(update)
