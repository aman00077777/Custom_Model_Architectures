"""
Utility functions for activation layers.
"""

import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    """
    Return a PyTorch activation module by name.

    Args:
        name: Activation name.

    Returns:
        nn.Module activation.

    Raises:
        ValueError: If activation is unsupported.
    """
    name = name.lower()

    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "softplus": nn.Softplus(),
        "identity": nn.Identity(),
    }

    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")

    return activations[name]