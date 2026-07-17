"""
Scaled dot-product Self-Attention module.

Usage:
    from fusion.models.custom.attention import SelfAttention
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention.

    Projects input ``x`` into queries, keys, and values via three separate
    linear layers (no bias), computes scaled dot-product attention scores,
    optionally applies causal masking, and returns the attended output.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        dropout (float): Dropout probability applied to attention weights.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, embed_dim)``.
            mask: Optional boolean causal mask of shape ``(T, T)`` or
                ``(B, T, T)``.  Positions where the mask is ``True`` are
                filled with ``-inf`` before the softmax (attended-away).

        Returns:
            Output tensor of shape ``(B, T, embed_dim)``.
        """
        # (B, T, embed_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Attention scores: (B, T, T)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        if mask is not None:
            # Broadcast mask to (B, T, T) if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # (1, T, T)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output: (B, T, embed_dim)
        return torch.bmm(attn_weights, V)
