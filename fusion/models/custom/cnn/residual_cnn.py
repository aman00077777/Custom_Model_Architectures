"""
fusion/models/custom/cnn/residual_cnn.py

Defines **ResidualCNN** — a configurable stack of residual blocks, each
containing two Conv2d → BN → ReLU sequences plus a learnable skip
connection that applies a 1×1 convolution when the channel dimensions
differ.

Expected config keys
--------------------
in_channels : int
    Input image channels.
block_channels : List[int]
    Output channel count for each residual block (determines depth).
kernel_size : int, default 3
    Spatial kernel size for the 3×3 convolutions inside each block.
activation : str, default "relu"
    Activation function name.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from fusion.models.utils.activation import get_activation

Config = Union[Dict[str, Any], Any]


def _cfg(config: Config, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# ------------------------------------------------------------------
# Inner residual block
# ------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two-conv residual block with optional 1×1 projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = get_activation(activation)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = get_activation(activation)

        # 1×1 projection when channels change
        self.skip: nn.Module
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = self.act2(out)
        return out


# ------------------------------------------------------------------
# ResidualCNN
# ------------------------------------------------------------------

class ResidualCNN(nn.Module):
    """Stack of ``ResidualBlock`` modules → adaptive pool → flatten."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        in_channels: int = _cfg(config, "in_channels")
        block_channels: List[int] = _cfg(config, "block_channels")
        kernel_size: int = _cfg(config, "kernel_size", 3)
        activation: str = _cfg(config, "activation", "relu")

        blocks: List[nn.Module] = []
        ch_in = in_channels
        for ch_out in block_channels:
            blocks.append(
                ResidualBlock(ch_in, ch_out, kernel_size=kernel_size, activation=activation)
            )
            ch_in = ch_out

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self._output_dim: int = block_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W) → (B, feature_dim)``."""
        x = self.blocks(x)
        x = self.pool(x)
        return self.flatten(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim
