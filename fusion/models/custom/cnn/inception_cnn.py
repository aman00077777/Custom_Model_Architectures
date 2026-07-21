"""
fusion/models/custom/cnn/inception_cnn.py

Defines **InceptionCNN** — a GoogLeNet-inspired network that stacks
``InceptionModule`` blocks.  Each module has four parallel paths:

    1. 1×1 convolution
    2. 1×1 → 3×3 convolution
    3. 1×1 → 5×5 convolution
    4. 3×3 MaxPool → 1×1 convolution

Outputs are concatenated along the channel axis.

Expected config keys
--------------------
in_channels : int
    Input image channels.
num_blocks : int
    Number of InceptionModule blocks to stack.
ch_1x1 : int
    Output channels for path 1.
ch_reduce_3x3 : int
    Reduction channels (1×1) before the 3×3 conv in path 2.
ch_3x3 : int
    Output channels for path 2.
ch_reduce_5x5 : int
    Reduction channels (1×1) before the 5×5 conv in path 3.
ch_5x5 : int
    Output channels for path 3.
ch_pool_proj : int
    Output channels for the pool-projection path 4.
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
# Inception module
# ------------------------------------------------------------------

class InceptionModule(nn.Module):
    """Four-path inception block → channel concatenation."""

    def __init__(
        self,
        in_channels: int,
        ch_1x1: int,
        ch_reduce_3x3: int,
        ch_3x3: int,
        ch_reduce_5x5: int,
        ch_5x5: int,
        ch_pool_proj: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        # Path 1: 1×1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch_1x1, kernel_size=1),
            nn.BatchNorm2d(ch_1x1),
            get_activation(activation),
        )

        # Path 2: 1×1 → 3×3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch_reduce_3x3, kernel_size=1),
            nn.BatchNorm2d(ch_reduce_3x3),
            get_activation(activation),
            nn.Conv2d(ch_reduce_3x3, ch_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_3x3),
            get_activation(activation),
        )

        # Path 3: 1×1 → 5×5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch_reduce_5x5, kernel_size=1),
            nn.BatchNorm2d(ch_reduce_5x5),
            get_activation(activation),
            nn.Conv2d(ch_reduce_5x5, ch_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch_5x5),
            get_activation(activation),
        )

        # Path 4: 3×3 MaxPool → 1×1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, ch_pool_proj, kernel_size=1),
            nn.BatchNorm2d(ch_pool_proj),
            get_activation(activation),
        )

        self._out_channels = ch_1x1 + ch_3x3 + ch_5x5 + ch_pool_proj

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


# ------------------------------------------------------------------
# InceptionCNN
# ------------------------------------------------------------------

class InceptionCNN(nn.Module):
    """Stack of ``InceptionModule`` blocks → adaptive pool → flatten."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        in_channels: int = _cfg(config, "in_channels")
        num_blocks: int = _cfg(config, "num_blocks")
        ch_1x1: int = _cfg(config, "ch_1x1")
        ch_reduce_3x3: int = _cfg(config, "ch_reduce_3x3")
        ch_3x3: int = _cfg(config, "ch_3x3")
        ch_reduce_5x5: int = _cfg(config, "ch_reduce_5x5")
        ch_5x5: int = _cfg(config, "ch_5x5")
        ch_pool_proj: int = _cfg(config, "ch_pool_proj")
        activation: str = _cfg(config, "activation", "relu")

        blocks: List[nn.Module] = []
        ch_in = in_channels

        for _ in range(num_blocks):
            block = InceptionModule(
                in_channels=ch_in,
                ch_1x1=ch_1x1,
                ch_reduce_3x3=ch_reduce_3x3,
                ch_3x3=ch_3x3,
                ch_reduce_5x5=ch_reduce_5x5,
                ch_5x5=ch_5x5,
                ch_pool_proj=ch_pool_proj,
                activation=activation,
            )
            blocks.append(block)
            ch_in = block.out_channels  # feed concat width to next block

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self._output_dim: int = blocks[-1].out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W) → (B, feature_dim)``."""
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.flatten(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim
