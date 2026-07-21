"""
fusion/models/custom/mlp/residual_mlp.py

Defines ResidualMLP: stacked two-layer residual blocks (ResNet-style,
for vectors).

Expected config parameters:
    input_dim (int)
    hidden_dim (int): Shared width for input projection and every
        residual block.
    num_blocks (int): Number of residual blocks to stack.
    output_dim (int)
    activation (str): Default "relu".
"""

import torch
import torch.nn as nn

from fusion.models.utils.activation import get_activation


class _ResidualBlock(nn.Module):
    """
    Residual block:

    output = activation(linear2(activation(linear1(x)))) + x
    """

    def __init__(
        self,
        dim: int,
        activation: str,
    ):
        super().__init__()

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, dim).

        Returns:
            Tensor of shape (B, dim).
        """
        out = self.activation(self.linear1(x))
        out = self.activation(self.linear2(out))

        return out + x


class ResidualMLP(nn.Module):
    """
    Optional input projection followed by residual blocks,
    then a final output head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        output_dim: int,
        activation: str = "relu",
    ):
        """
        Args:
            input_dim: Dimensionality of the input.
            hidden_dim: Width shared by every residual block.
            num_blocks: Number of residual blocks to stack.
            output_dim: Dimensionality of the output.
            activation: Activation used inside each residual block.
        """
        super().__init__()

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.blocks = nn.ModuleList(
            [
                _ResidualBlock(hidden_dim, activation)
                for _ in range(num_blocks)
            ]
        )

        self.output_head = nn.Linear(
            hidden_dim,
            output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, input_dim).

        Returns:
            Tensor of shape (B, output_dim).
        """
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return self.output_head(x)

    @classmethod
    def from_config(cls, config) -> "ResidualMLP":
        """
        Build a ResidualMLP from a Config object.
        """
        return cls(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            output_dim=config.output_dim,
            activation=config.get("activation", "relu"),
        )