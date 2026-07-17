"""
Spatial Attention module for 4-D feature maps.

Usage:
    from fusion.models.custom.attention import SpatialAttention
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fusion.models.custom.attention.self_attention import SelfAttention


class SpatialAttention(nn.Module):
    """Self-attention over the spatial dimensions of a 4-D feature map.

    Pipeline:
        1. Receive input ``(B, C, H, W)``.
        2. Flatten spatial dims → ``(B, C, H*W)``.
        3. Transpose to ``(B, H*W, C)`` — treat spatial positions as tokens.
        4. Apply :class:`SelfAttention` along the spatial (token) dimension.
        5. Transpose back to ``(B, C, H*W)``.
        6. Reshape to ``(B, C, H, W)``.

    Args:
        channels (int): Number of channels ``C`` in the feature map, used as
            ``embed_dim`` for the internal :class:`SelfAttention`.
        dropout (float): Dropout probability forwarded to :class:`SelfAttention`.
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        # ``channels`` acts as embed_dim since each spatial token is a C-vector
        self.attn = SelfAttention(embed_dim=channels, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: 4-D feature map of shape ``(B, C, H, W)``.
            mask: Optional causal mask forwarded to :class:`SelfAttention`.

        Returns:
            Feature map of identical shape ``(B, C, H, W)``.
        """
        B, C, H, W = x.shape

        # Flatten spatial dims: (B, C, H*W)
        x_flat = x.view(B, C, H * W)

        # Treat spatial positions as sequence tokens: (B, H*W, C)
        x_seq = x_flat.transpose(1, 2)

        # Apply self-attention over the spatial dimension
        x_attended = self.attn(x_seq, mask=mask)  # (B, H*W, C)

        # Restore original spatial layout
        x_out = x_attended.transpose(1, 2).contiguous()  # (B, C, H*W)
        return x_out.view(B, C, H, W)                    # (B, C, H, W)
