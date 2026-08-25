"""Drop-in QKNorm attention used by the final SHARP comparison suite."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QKNormMultiheadAttention(nn.Module):
    """Batch-first multi-head attention with L2-normalized queries and keys."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
        **_: object,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if add_bias_kv or add_zero_attn:
            raise ValueError("SHARP does not use add_bias_kv or add_zero_attn")
        if not batch_first:
            raise ValueError("Final-suite QKNorm requires batch_first=True")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = float(dropout)
        self.batch_first = True
        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(kdim or embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(vdim or embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.logit_scale = nn.Parameter(
            torch.full(
                (num_heads,), math.log(math.sqrt(self.head_dim)), **factory_kwargs
            )
        )

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, length, _ = value.shape
        return value.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    @staticmethod
    def _apply_mask(logits: Tensor, mask: Tensor) -> Tensor:
        if mask.dtype == torch.bool:
            return logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits + mask.to(dtype=logits.dtype)

    def _merge_masks(
        self,
        logits: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool,
    ) -> Tensor:
        batch, heads, query_len, key_len = logits.shape
        if is_causal and attn_mask is None:
            attn_mask = torch.ones(
                query_len,
                key_len,
                dtype=torch.bool,
                device=logits.device,
            ).triu(diagonal=1)

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                expanded = attn_mask[None, None, :, :]
            elif attn_mask.ndim == 3 and attn_mask.shape[0] == batch * heads:
                expanded = attn_mask.view(batch, heads, query_len, key_len)
            elif attn_mask.ndim == 3 and attn_mask.shape[0] == batch:
                expanded = attn_mask[:, None, :, :]
            elif attn_mask.ndim == 4:
                expanded = attn_mask
            else:
                raise ValueError(
                    f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}"
                )
            logits = self._apply_mask(logits, expanded.to(device=logits.device))

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, key_len):
                raise ValueError(
                    "key_padding_mask must have shape "
                    f"{(batch, key_len)}, got {tuple(key_padding_mask.shape)}"
                )
            padding = key_padding_mask[:, None, None, :].to(device=logits.device)
            logits = self._apply_mask(logits, padding)
        return logits

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        **_: object,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        q = F.normalize(self._split_heads(self.q_proj(query)), dim=-1, eps=1e-6)
        k = F.normalize(self._split_heads(self.k_proj(key)), dim=-1, eps=1e-6)
        v = self._split_heads(self.v_proj(value))
        scale = self.logit_scale.exp().clamp(max=100.0).view(1, -1, 1, 1)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale
        logits = self._merge_masks(
            logits, attn_mask, key_padding_mask, is_causal
        )
        weights = torch.softmax(logits, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(
            query.shape[0], query.shape[1], self.embed_dim
        )
        output = self.out_proj(output)

        if not need_weights:
            return output, None
        if average_attn_weights:
            return output, weights.mean(dim=1)
        return output, weights
