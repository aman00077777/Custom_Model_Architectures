"""
fusion/models/custom/cnn/lightweight_cnn.py

Defines **LightweightCNN** — a MobileNet-style network built from
depthwise separable convolutions (depthwise Conv2d + pointwise 1×1
Conv2d), configurable via a simple channel/stride specification.

Expected config keys
--------------------
in_channels : int
    Input image channels.
block_channels : List[int]
    Output channels for each depthwise-separable block.
kernel_size : int, default 3
    Spatial kernel size for the depthwise convolution.
strides : List[int] | None, default None
    Stride for each block; defaults to 1 everywhere.
activation : str, default "relu"
    Activation function name.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from fusion.models.utils.activation import get_activation

Config = Union[Dict[str, Any], Any]


def _cfg(config: Config, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# ------------------------------------------------------------------
# Depthwise separable block
# ------------------------------------------------------------------

class DepthwiseSeparableBlock(nn.Module):
    """Depthwise Conv2d → BN → Act → Pointwise Conv2d → BN → Act."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            get_activation(activation),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


# ------------------------------------------------------------------
# LightweightCNN
# ------------------------------------------------------------------

class LightweightCNN(nn.Module):
    """Stack of ``DepthwiseSeparableBlock`` → adaptive pool → flatten."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        in_channels: int = _cfg(config, "in_channels")
        block_channels: List[int] = _cfg(config, "block_channels")
        kernel_size: int = _cfg(config, "kernel_size", 3)
        strides: Optional[List[int]] = _cfg(config, "strides", None)
        activation: str = _cfg(config, "activation", "relu")

        if strides is None:
            strides = [1] * len(block_channels)
        assert len(strides) == len(block_channels)

        blocks: List[nn.Module] = []
        ch_in = in_channels
        for ch_out, s in zip(block_channels, strides):
            blocks.append(
                DepthwiseSeparableBlock(
                    ch_in, ch_out,
                    kernel_size=kernel_size,
                    stride=s,
                    activation=activation,
                )
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
