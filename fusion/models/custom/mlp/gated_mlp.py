"""
fusion/models/custom/mlp/gated_mlp.py

Defines GatedMLP: stacked SwiGLU-style Gated Linear Units.

Expected config parameters:
    input_dim (int)
    hidden_dims (List[int])
    output_dim (int)
    activation (str): Activation applied on the value path,
        default "silu".
"""

from typing import List

import torch
import torch.nn as nn

from fusion.models.utils.activation import get_activation


class _GatedLayer(nn.Module):
    """
    One SwiGLU-style block:

    sigmoid(gate_proj(x)) * activation(value_proj(x))
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(in_dim, out_dim)
        self.value_proj = nn.Linear(in_dim, out_dim)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, in_dim).

        Returns:
            Tensor of shape (B, out_dim).
        """
        gate = torch.sigmoid(self.gate_proj(x))
        value = self.activation(self.value_proj(x))

        return gate * value


class GatedMLP(nn.Module):
    """
    A stack of gated (SwiGLU-style) layers ending in
    a plain Linear output head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "silu",
    ):
        """
        Args:
            input_dim: Dimensionality of the input.
            hidden_dims: Widths of each gated hidden layer.
            output_dim: Dimensionality of the output.
            activation: Activation used on the value path
                of every gated layer.
        """
        super().__init__()

        dims = [input_dim] + list(hidden_dims)

        self.gated_layers = nn.ModuleList(
            [
                _GatedLayer(
                    dims[i],
                    dims[i + 1],
                    activation,
                )
                for i in range(len(dims) - 1)
            ]
        )

        self.output_head = nn.Linear(
            dims[-1],
            output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, input_dim).

        Returns:
            Tensor of shape (B, output_dim).
        """
        for layer in self.gated_layers:
            x = layer(x)

        return self.output_head(x)

    @classmethod
    def from_config(cls, config) -> "GatedMLP":
        """
        Build a GatedMLP from a Config object.
        """
        return cls(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            activation=config.get("activation", "silu"),
        )