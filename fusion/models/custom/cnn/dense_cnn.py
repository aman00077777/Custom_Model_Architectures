"""
fusion/models/custom/cnn/dense_cnn.py

Defines **DenseCNN** — a DenseNet-style network where each layer inside a
dense block receives the **concatenation of all preceding feature maps**.
Transition layers between dense blocks reduce channel count (1×1 conv)
and spatial resolution (2×2 avg-pool).

Expected config keys
--------------------
in_channels : int
    Input image channels.
growth_rate : int
    Number of new channels produced by each dense layer.
num_layers_per_block : List[int]
    Number of dense layers in each dense block.
compression : float, default 0.5
    Compression factor for transition layers (0 < θ ≤ 1).
activation : str, default "relu"
    Activation function name.
"""

from __future__ import annotations

import math
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
# Dense layer  (BN → Act → Conv)
# ------------------------------------------------------------------

class _DenseLayer(nn.Module):
    """Single BN → Activation → 3×3 Conv layer that receives concatenated
    feature maps from all previous layers."""

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            get_activation(activation),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layer(x)
        return torch.cat([x, new_features], dim=1)


# ------------------------------------------------------------------
# Dense block
# ------------------------------------------------------------------

class DenseBlock(nn.Module):
    """Stack of ``_DenseLayer`` modules with channel concatenation."""

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_channels
        for _ in range(num_layers):
            layers.append(_DenseLayer(ch, growth_rate, activation))
            ch += growth_rate
        self.layers = nn.Sequential(*layers)
        self._out_channels = ch

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ------------------------------------------------------------------
# Transition layer
# ------------------------------------------------------------------

class _Transition(nn.Module):
    """1×1 conv (channel reduction) → 2×2 avg-pool (spatial halving)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            get_activation(activation),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


# ------------------------------------------------------------------
# DenseCNN
# ------------------------------------------------------------------

class DenseCNN(nn.Module):
    """Stack of ``DenseBlock + Transition`` → adaptive pool → flatten."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        in_channels: int = _cfg(config, "in_channels")
        growth_rate: int = _cfg(config, "growth_rate")
        num_layers_per_block: List[int] = _cfg(config, "num_layers_per_block")
        compression: float = _cfg(config, "compression", 0.5)
        activation: str = _cfg(config, "activation", "relu")

        assert 0.0 < compression <= 1.0, "compression must be in (0, 1]"

        features: List[nn.Module] = []
        ch = in_channels

        for i, num_layers in enumerate(num_layers_per_block):
            block = DenseBlock(ch, num_layers, growth_rate, activation)
            features.append(block)
            ch = block.out_channels

            # Transition after every block except the last
            if i < len(num_layers_per_block) - 1:
                out_ch = int(math.floor(ch * compression))
                features.append(_Transition(ch, out_ch, activation))
                ch = out_ch

        # Final BN
        features.append(nn.BatchNorm2d(ch))

        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self._output_dim: int = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W) → (B, feature_dim)``."""
        x = self.features(x)
        x = self.pool(x)
        return self.flatten(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim
