"""
Cross-Attention module.

Usage:
    from fusion.models.custom.attention import CrossAttention
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Scaled dot-product cross-attention.

    Queries come from one sequence; keys and values come from another.
    The output length matches the query sequence length.

    Args:
        query_dim (int): Dimensionality of the query input.
        kv_dim (int): Dimensionality of the key/value input.
        out_dim (int): Output projection dimensionality. Defaults to
            ``query_dim`` if not specified.
        dropout (float): Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        # Attention is computed in query_dim space
        self.scale = math.sqrt(query_dim)

        # Q is projected from query; K, V from key_value
        self.W_q = nn.Linear(query_dim, query_dim, bias=False)
        self.W_k = nn.Linear(kv_dim, query_dim, bias=False)
        self.W_v = nn.Linear(kv_dim, self.out_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Query tensor of shape ``(B, T_q, query_dim)``.
            key_value: Context tensor of shape ``(B, T_kv, kv_dim)``.
            mask: Optional boolean mask of shape ``(B, T_q, T_kv)`` or
                ``(T_q, T_kv)``.  ``True`` positions are set to ``-inf``.

        Returns:
            Output tensor of shape ``(B, T_q, out_dim)``.
        """
        # Projections
        Q = self.W_q(query)       # (B, T_q, query_dim)
        K = self.W_k(key_value)   # (B, T_kv, query_dim)
        V = self.W_v(key_value)   # (B, T_kv, out_dim)

        # Attention scores: (B, T_q, T_kv)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output aligned with query length: (B, T_q, out_dim)
        return torch.bmm(attn_weights, V)
