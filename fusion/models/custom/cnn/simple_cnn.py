"""
fusion/models/custom/cnn/simple_cnn.py

Defines **SimpleCNN** — a straightforward stack of Conv2d layers built
entirely from a ``config`` specification.

Expected config keys
--------------------
in_channels : int
    Number of channels in the input image (e.g. 3 for RGB).
out_channels : List[int]
    Number of output channels for each convolutional layer.
kernel_sizes : List[int]
    Kernel size for each convolutional layer.
strides : List[int]
    Stride for each convolutional layer.
use_batchnorm : bool, default True
    Whether to insert BatchNorm2d after each Conv2d.
dropout_rate : float, default 0.0
    Spatial dropout probability applied after each activation.
activation : str, default "relu"
    Activation function name understood by ``get_activation``.
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


class SimpleCNN(nn.Module):
    """Sequential ``Conv2d → [BN] → Activation → [Dropout]`` stack.

    Ends with ``AdaptiveAvgPool2d((1,1))`` and ``Flatten`` so the output
    is always ``(B, feature_dim)`` regardless of spatial input size.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        in_channels: int = _cfg(config, "in_channels")
        out_channels: List[int] = _cfg(config, "out_channels")
        kernel_sizes: List[int] = _cfg(config, "kernel_sizes")
        strides: List[int] = _cfg(config, "strides")
        use_batchnorm: bool = _cfg(config, "use_batchnorm", True)
        dropout_rate: float = _cfg(config, "dropout_rate", 0.0)
        activation: str = _cfg(config, "activation", "relu")

        assert len(out_channels) == len(kernel_sizes) == len(strides), (
            "out_channels, kernel_sizes and strides must have the same length."
        )

        layers: List[nn.Module] = []
        ch_in = in_channels

        for ch_out, ks, s in zip(out_channels, kernel_sizes, strides):
            padding = ks // 2  # same-style padding
            layers.append(
                nn.Conv2d(ch_in, ch_out, kernel_size=ks, stride=s, padding=padding)
            )
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(ch_out))
            layers.append(get_activation(activation))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout2d(dropout_rate))
            ch_in = ch_out

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())

        self.net = nn.Sequential(*layers)
        self._output_dim: int = out_channels[-1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W) → (B, feature_dim)``."""
        return self.net(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim
