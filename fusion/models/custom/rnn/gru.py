"""
fusion/models/custom/rnn/gru.py

Defines GRUEncoder: a config-driven wrapper around nn.GRU (no cell state).

Expected config parameters:
    input_size (int)
    hidden_size (int)
    num_layers (int)
    dropout (float): Dropout between GRU layers
        (ignored if num_layers == 1).
    return_all_states (bool): If True, forward returns every time
        step's output; if False, forward returns only the final
        hidden state.
"""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """
    Thin wrapper around nn.GRU exposing a last-state or
    all-states output mode.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        return_all_states: bool = False,
    ):
        """
        Args:
            input_size: Number of expected features in the input.
            hidden_size: Number of features in the hidden state.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability between layers.
                nn.GRU ignores this if num_layers == 1.
            return_all_states: Output mode flag (see forward()).
        """
        super().__init__()

        self.return_all_states = return_all_states

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, input_size).

        Returns:
            (B, T, hidden_size) if return_all_states is True,
            otherwise (B, hidden_size).
        """
        outputs, h_n = self.gru(x)  # GRU has no cell state.

        if self.return_all_states:
            return outputs

        return h_n[-1]

    @classmethod
    def from_config(cls, config) -> "GRUEncoder":
        """
        Build a GRUEncoder from a Config object.
        """
        return cls(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.0),
            return_all_states=config.get("return_all_states", False),
        )