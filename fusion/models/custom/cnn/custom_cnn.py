"""
fusion/models/custom/cnn/custom_cnn.py

Defines **CustomCNN** — a fully dynamic, config-driven convolutional
network built by iterating a ``layers`` list specification.

Each entry in the list is a dict with a ``type`` key that selects the
layer kind and additional kwargs:

Supported layer types
---------------------
``conv``
    Conv2d.  Keys: ``out_channels``, ``kernel_size``, ``stride`` (default 1).
``batchnorm``
    BatchNorm2d.  (infers channels automatically.)
``activation``
    Activation by name via ``get_activation``.  Keys: ``name`` (default "relu").
``dropout``
    Spatial dropout.  Keys: ``rate`` (default 0.5).
``maxpool``
    MaxPool2d.  Keys: ``kernel_size`` (default 2), ``stride`` (default 2).
``avgpool``
    AvgPool2d.  Keys: ``kernel_size`` (default 2), ``stride`` (default 2).

Expected config keys
--------------------
in_channels : int
    Input image channels.
layers : List[Dict[str, Any]]
    Ordered layer specification list.
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


class CustomCNN(nn.Module):
    """Dynamically assembled CNN from a config ``layers`` specification.

    Ends with ``AdaptiveAvgPool2d((1,1))`` and ``Flatten`` so the
    output is always ``(B, feature_dim)``.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()

        in_channels: int = _cfg(config, "in_channels")
        layer_specs: List[Dict[str, Any]] = _cfg(config, "layers")

        modules: List[nn.Module] = []
        ch = in_channels

        for spec in layer_specs:
            layer_type = spec["type"]

            if layer_type == "conv":
                out_ch = spec["out_channels"]
                ks = spec["kernel_size"]
                stride = spec.get("stride", 1)
                padding = ks // 2
                modules.append(
                    nn.Conv2d(ch, out_ch, kernel_size=ks,
                              stride=stride, padding=padding)
                )
                ch = out_ch

            elif layer_type == "batchnorm":
                modules.append(nn.BatchNorm2d(ch))

            elif layer_type == "activation":
                name = spec.get("name", "relu")
                modules.append(get_activation(name))

            elif layer_type == "dropout":
                rate = spec.get("rate", 0.5)
                modules.append(nn.Dropout2d(rate))

            elif layer_type == "maxpool":
                ks = spec.get("kernel_size", 2)
                stride = spec.get("stride", 2)
                modules.append(nn.MaxPool2d(kernel_size=ks, stride=stride))

            elif layer_type == "avgpool":
                ks = spec.get("kernel_size", 2)
                stride = spec.get("stride", 2)
                modules.append(nn.AvgPool2d(kernel_size=ks, stride=stride))

            else:
                raise ValueError(f"Unsupported layer type: {layer_type!r}")

        # Final adaptive pool + flatten
        modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        modules.append(nn.Flatten())

        self.net = nn.Sequential(*modules)
        self._output_dim: int = ch
        self._num_config_layers: int = len(layer_specs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W) → (B, feature_dim)``."""
        return self.net(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def num_config_layers(self) -> int:
        """Number of layers parsed from the config specification."""
        return self._num_config_layers
