"""
fusion.models.utils.activation

Temporary activation utility used until the Phase 4 implementation
is available.
"""

import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Return an activation module given its name.

    Args:
        name: Name of the activation function.

    Returns:
        Instantiated PyTorch activation module.

    Raises:
        ValueError: If the activation name is not supported.
    """
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }

    key = name.lower()

    if key not in mapping:
        raise ValueError(
            f"Unknown activation: {name}"
        )

    return mapping[key]()