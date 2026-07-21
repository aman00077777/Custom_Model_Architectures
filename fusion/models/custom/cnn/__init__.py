"""
fusion/models/custom/cnn/__init__.py

Common interface and registry for all CNN building blocks.

Every CNN class in this package:
    1. Accepts a ``config`` dict/object in ``__init__``.
    2. Exposes ``forward(x: Tensor) -> Tensor`` mapping
       (B, C, H, W) -> (B, feature_dim).
    3. Stores the final feature dimensionality in ``output_dim``.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

Config = Union[Dict[str, Any], Any]
"""Accepted config type: plain dict **or** an object with attribute access."""


def _cfg_get(config: Config, key: str, default: Any = None) -> Any:
    """Retrieve *key* from a dict-like or attribute-bearing config."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


# ---------------------------------------------------------------------------
# Base mixin
# ---------------------------------------------------------------------------

class BaseCNN(nn.Module, abc.ABC):
    """Abstract base for every CNN block in the Fusion framework.

    Subclasses **must**:
    * Call ``super().__init__()`` in their constructor.
    * Set ``self._output_dim: int`` before ``forward`` is ever invoked.
    * Implement ``forward(x)`` returning a 2-D tensor ``(B, feature_dim)``.
    """

    _output_dim: int

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, C, H, W) -> (B, feature_dim)."""
        ...

    @property
    def output_dim(self) -> int:
        """Dimensionality of the final 1-D feature vector."""
        return self._output_dim


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

from fusion.models.custom.cnn.simple_cnn import SimpleCNN          # noqa: E402
from fusion.models.custom.cnn.residual_cnn import ResidualCNN      # noqa: E402
from fusion.models.custom.cnn.multi_scale_cnn import MultiScaleCNN  # noqa: E402
from fusion.models.custom.cnn.lightweight_cnn import LightweightCNN  # noqa: E402
from fusion.models.custom.cnn.inception_cnn import InceptionCNN    # noqa: E402
from fusion.models.custom.cnn.dense_cnn import DenseCNN            # noqa: E402
from fusion.models.custom.cnn.custom_cnn import CustomCNN          # noqa: E402

__all__ = [
    "BaseCNN",
    "SimpleCNN",
    "ResidualCNN",
    "MultiScaleCNN",
    "LightweightCNN",
    "InceptionCNN",
    "DenseCNN",
    "CustomCNN",
]
