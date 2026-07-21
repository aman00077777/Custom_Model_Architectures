"""
fusion.models.custom.mlp

Configurable, from-scratch MLP architectures:

- SimpleMLP
- DeepMLP
- GatedMLP
- ResidualMLP
- MixtureOfExperts
- AdaptiveMLP

All classes expose a `from_config(config)` classmethod in addition
to their standard constructors.
"""

from fusion.models.custom.mlp.adaptive_mlp import AdaptiveMLP
from fusion.models.custom.mlp.deep_mlp import DeepMLP
from fusion.models.custom.mlp.gated_mlp import GatedMLP
from fusion.models.custom.mlp.mixture_of_experts import MixtureOfExperts
from fusion.models.custom.mlp.residual_mlp import ResidualMLP
from fusion.models.custom.mlp.simple_mlp import SimpleMLP

__all__ = [
    "SimpleMLP",
    "DeepMLP",
    "GatedMLP",
    "ResidualMLP",
    "MixtureOfExperts",
    "AdaptiveMLP",
]