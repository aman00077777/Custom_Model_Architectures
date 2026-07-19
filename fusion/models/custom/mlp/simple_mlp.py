"""
fusion/models/custom/mlp/simple_mlp.py

Defines SimpleMLP: a configurable feed-forward network.

Expected config parameters:
    input_dim (int): Size of the input feature vector.
    hidden_dims (List[int]): Sizes of each hidden layer, e.g. [128, 64].
    output_dim (int): Size of the output feature vector.
    activation (str): Activation function name, default "relu".
    use_batchnorm (bool): Apply BatchNorm1d after each hidden Linear,
        default False.
    dropout_rate (float): Dropout probability after each hidden activation,
        default 0.0.
"""

from typing import List

import torch
import torch.nn as nn

from fusion.models.utils.activation import get_activation


class SimpleMLP(nn.Module):
    """
    Feed-forward network:

    (Linear -> BatchNorm? -> Activation -> Dropout?) * N -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        use_batchnorm: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
        Args:
            input_dim: Dimensionality of the input features.
            hidden_dims: List of hidden layer widths.
                Can be empty for a single Linear layer.
            output_dim: Dimensionality of the output.
            activation: Name understood by get_activation.
            use_batchnorm: Whether to insert BatchNorm1d
                after hidden layers.
            dropout_rate: Dropout probability applied
                after hidden activations.
        """
        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers = []

        for i in range(len(dims) - 1):
            is_last = i == len(dims) - 2

            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if not is_last:
                if use_batchnorm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

                layers.append(get_activation(activation))

                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, input_dim).

        Returns:
            Tensor of shape (B, output_dim).
        """
        return self.net(x)

    @classmethod
    def from_config(cls, config) -> "SimpleMLP":
        """
        Build a SimpleMLP from a Config object.
        """
        return cls(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            activation=config.get("activation", "relu"),
            use_batchnorm=config.get("use_batchnorm", False),
            dropout_rate=config.get("dropout_rate", 0.0),
        ) 