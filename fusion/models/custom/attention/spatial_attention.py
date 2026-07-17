import torch
import torch.nn as nn
from .self_attention import SelfAttention


class SpatialAttention(nn.Module):
    """
    SpatialAttention module that flattens spatial dimensions of 4D inputs,
    applies SelfAttention along the spatial dimension, and restores 4D shape.
    """
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.spatial_attn = SelfAttention(embed_dim=channels, dropout=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions to (B, C, H*W) and transpose to (B, H*W, C)
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        
        # Apply SelfAttention
        out_flat = self.spatial_attn(x_flat)  # (B, H*W, C)
        
        # Transpose to (B, C, H*W) and reshape back to (B, C, H, W)
        out = out_flat.transpose(1, 2).view(B, C, H, W)
        return out
