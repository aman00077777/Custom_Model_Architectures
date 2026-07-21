"""
fusion/models/custom/mlp/deep_mlp.py

Defines DeepMLP: SimpleMLP with a uniform hidden width repeated
num_layers times.

Expected config parameters:
    input_dim (int)
    hidden_dim (int): Width used for every hidden layer.
    num_layers (int): How many hidden layers to stack.
    output_dim (int)
    activation (str): Same as SimpleMLP.
    use_batchnorm (bool): Same as SimpleMLP.
    dropout_rate (float): Same as SimpleMLP.
"""

import torch
import torch.nn as nn

from fusion.models.custom.mlp.simple_mlp import SimpleMLP


class DeepMLP(nn.Module):
    """
    SimpleMLP variant configured with a single hidden width
    repeated num_layers times.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        activation: str = "relu",
        use_batchnorm: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
        Args:
            input_dim: Dimensionality of the input.
            hidden_dim: Width shared by every hidden layer.
            num_layers: Number of hidden layers to stack.
            output_dim: Dimensionality of the output.
            activation: Name understood by get_activation.
            use_batchnorm: Whether to insert BatchNorm1d
                after hidden layers.
            dropout_rate: Dropout probability applied
                after hidden activations.
        """
        super().__init__()

        self.mlp = SimpleMLP(
            input_dim=input_dim,
            hidden_dims=[hidden_dim] * num_layers,
            output_dim=output_dim,
            activation=activation,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, input_dim).

        Returns:
            Tensor of shape (B, output_dim).
        """
        return self.mlp(x)

    @classmethod
    def from_config(cls, config) -> "DeepMLP":
        """
        Build a DeepMLP from a Config object.
        """
        return cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_dim=config.output_dim,
            activation=config.get("activation", "relu"),
            use_batchnorm=config.get("use_batchnorm", False),
            dropout_rate=config.get("dropout_rate", 0.0),
        )