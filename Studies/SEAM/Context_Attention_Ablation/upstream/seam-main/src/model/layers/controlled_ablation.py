import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class QKNormMultiheadAttention(nn.Module):
    """Batch-first multi-head attention with per-head query/key normalisation."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        batch_first: bool = True,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not batch_first:
            raise ValueError("QKNormMultiheadAttention requires batch_first=True")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.logit_scale = nn.Parameter(
            torch.full((num_heads,), math.log(math.sqrt(self.head_dim)))
        )

    def _heads(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _apply_attention_mask(logits: Tensor, mask: Tensor) -> Tensor:
        batch, heads, query_len, key_len = logits.shape
        if mask.ndim == 2:
            mask = mask.view(1, 1, query_len, key_len)
        elif mask.ndim == 3:
            if mask.shape[0] == batch * heads:
                mask = mask.view(batch, heads, query_len, key_len)
            elif mask.shape[0] == batch:
                mask = mask.view(batch, 1, query_len, key_len)
            else:
                raise ValueError(f"Unsupported 3-D attention mask shape: {mask.shape}")
        elif mask.ndim != 4:
            raise ValueError(f"Unsupported attention mask rank: {mask.ndim}")

        mask = mask.to(device=logits.device)
        if mask.dtype == torch.bool:
            return logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits + mask.to(dtype=logits.dtype)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        **_: object,
    ):
        q = F.normalize(self._heads(self.q_proj(query)), dim=-1, eps=1e-6)
        k = F.normalize(self._heads(self.k_proj(key)), dim=-1, eps=1e-6)
        v = self._heads(self.v_proj(value))

        scale = self.logit_scale.exp().clamp(max=100.0).view(1, -1, 1, 1)
        logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            logits = self._apply_attention_mask(logits, attn_mask)
        if key_padding_mask is not None:
            invalid = key_padding_mask.to(torch.bool)[:, None, None, :]
            logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)

        weights = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        weights = torch.nan_to_num(weights, nan=0.0)
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


def make_attention(
    variant: str,
    dim: int,
    num_heads: int,
    qkv_bias: bool,
    attn_drop: float,
    kdim: Optional[int] = None,
    vdim: Optional[int] = None,
) -> nn.Module:
    if variant == "qknorm":
        return QKNormMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            # SEAM passes qkv_bias to MultiheadAttention.add_bias_kv; its
            # projection biases retain PyTorch's default True setting.
            bias=True,
            batch_first=True,
            kdim=kdim,
            vdim=vdim,
        )
    return nn.MultiheadAttention(
        dim,
        num_heads=num_heads,
        add_bias_kv=qkv_bias,
        dropout=attn_drop,
        batch_first=True,
        kdim=kdim,
        vdim=vdim,
    )


class RelativeGeometryBias(nn.Module):
    """Learned per-head scene-attention bias from relative pose geometry."""

    def __init__(self, num_heads: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.network = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        centers: Tensor,
        headings: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        relative_xy = centers[:, None, :, :] - centers[:, :, None, :]
        distance = torch.linalg.vector_norm(relative_xy, dim=-1, keepdim=True)
        heading_delta = headings[:, None, :] - headings[:, :, None]
        features = torch.cat(
            (
                relative_xy / 30.0,
                distance / 30.0,
                torch.cos(heading_delta).unsqueeze(-1),
                torch.sin(heading_delta).unsqueeze(-1),
            ),
            dim=-1,
        )
        bias = self.network(features).permute(0, 3, 1, 2).contiguous()
        invalid_keys = (~valid_mask.to(torch.bool))[:, None, None, :]
        bias = bias.masked_fill(invalid_keys, torch.finfo(bias.dtype).min)
        batch, heads, query_len, key_len = bias.shape
        return bias.view(batch * heads, query_len, key_len)


class UncertaintyTargetContext(nn.Module):
    """Maps previous-window modal uncertainty to target radius and feature gain."""

    def __init__(self, base_radius: float = 30.0) -> None:
        super().__init__()
        self.base_radius = base_radius
        self.controller = nn.Linear(3, 2)
        nn.init.zeros_(self.controller.weight)
        nn.init.zeros_(self.controller.bias)

    def forward(self, logits: Tensor):
        probabilities = torch.softmax(logits.float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / math.log(probabilities.shape[-1])
        entropy = entropy.unsqueeze(-1).expand_as(probabilities)
        inputs = torch.stack(
            (probabilities, 1.0 - probabilities, entropy), dim=-1
        )
        controls = self.controller(inputs.to(self.controller.weight.dtype))
        radius = self.base_radius + 10.0 * torch.tanh(controls[..., 0])
        gain = 1.0 + 0.25 * torch.tanh(controls[..., 1])
        return radius.to(logits.dtype), gain.to(logits.dtype)

    def zero_dependency(self, reference: Tensor) -> Tensor:
        zero = sum(parameter.sum() for parameter in self.parameters()) * 0.0
        return zero.to(dtype=reference.dtype, device=reference.device)
