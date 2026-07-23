"""
fusion/models/custom/cnn/multi_scale_cnn.py

Defines **MultiScaleCNN** — a multi-scale feature extractor that runs
parallel Conv2d branches with different kernel sizes, concatenates the
outputs along the channel axis, and fuses them with a 1×1 convolution.

Expected config keys
--------------------
in_channels : int
    Input image channels.
branch_kernel_sizes : List[int]
    Kernel sizes for the parallel branches, e.g. ``[3, 5, 7]``.
branch_out_channels : int
    Number of output channels **per branch**.
output_channels : int
    Number of output channels after 1×1 fusion conv.
num_blocks : int, default 1
    Number of stacked multi-scale blocks.
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
# Single multi-scale block
# ------------------------------------------------------------------

class MultiScaleBlock(nn.Module):
    """Parallel branches → concat → 1×1 fusion → BN → activation."""

    def __init__(
        self,
        in_channels: int,
        branch_kernel_sizes: List[int],
        branch_out_channels: int,
        output_channels: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.branches = nn.ModuleList()
        for ks in branch_kernel_sizes:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_out_channels,
                          kernel_size=ks, padding=ks // 2),
                nn.BatchNorm2d(branch_out_channels),
                get_activation(activation),
            )
            self.branches.append(branch)

        concat_channels = branch_out_channels * len(branch_kernel_sizes)
        self.fuse = nn.Sequential(
            nn.Conv2d(concat_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            get_activation(activation),
        )

        self._concat_channels = concat_channels

    @property
    def concat_channels(self) -> int:
        """Channel count right after concatenation (before fusion)."""
        return self._concat_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)  # (B, C_concat, H, W)
        return self.fuse(concatenated)


# ------------------------------------------------------------------
# MultiScaleCNN
# ------------------------------------------------------------------

class MultiScaleCNN(nn.Module):
    """Stack of ``MultiScaleBlock`` modules → adaptive pool → flatten."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        in_channels: int = _cfg(config, "in_channels")
        branch_kernel_sizes: List[int] = _cfg(config, "branch_kernel_sizes")
        branch_out_channels: int = _cfg(config, "branch_out_channels")
        output_channels: int = _cfg(config, "output_channels")
        num_blocks: int = _cfg(config, "num_blocks", 1)
        activation: str = _cfg(config, "activation", "relu")

        blocks: List[nn.Module] = []
        ch_in = in_channels
        for _ in range(num_blocks):
            block = MultiScaleBlock(
                in_channels=ch_in,
                branch_kernel_sizes=branch_kernel_sizes,
                branch_out_channels=branch_out_channels,
                output_channels=output_channels,
                activation=activation,
            )
            blocks.append(block)
            ch_in = output_channels

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self._output_dim: int = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W) → (B, feature_dim)``."""
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.flatten(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim
