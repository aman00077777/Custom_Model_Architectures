"""
Multi-Head Attention module.

Usage:
    from fusion.models.custom.attention import MultiHeadAttention
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product self-attention.

    Projects inputs into Q, K, V, reshapes them into ``num_heads`` sub-spaces,
    computes independent attention per head, concatenates the results, and
    applies a final linear projection ``W_o``.

    Args:
        embed_dim (int): Total embedding dimensionality.
            Must be divisible by ``num_heads``.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability applied to attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"`embed_dim` ({embed_dim}) must be divisible by "
                f"`num_heads` ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Single fused projections for efficiency (split later)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, T, embed_dim)`` → ``(B, num_heads, T, head_dim)``."""
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B, H, T, head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, num_heads, T, head_dim)`` → ``(B, T, embed_dim)``."""
        B, H, T, head_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, T, H, head_dim)
        return x.view(B, T, H * head_dim)        # (B, T, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, T, embed_dim)``.
            mask: Optional causal mask of shape ``(T, T)`` or
                ``(B, num_heads, T, T)``.  ``True`` positions → ``-inf``.

        Returns:
            Output tensor of shape ``(B, T, embed_dim)``.
        """
        # Project and split into heads → (B, H, T, head_dim)
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        # Per-head scaled dot-product attention: (B, H, T, T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # Support (T, T) → expand to (1, 1, T, T)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: (B, H, T, head_dim)
        attended = torch.matmul(attn_weights, V)

        # Merge heads and project: (B, T, embed_dim)
        out = self._merge_heads(attended)
        return self.W_o(out)
