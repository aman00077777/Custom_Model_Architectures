"""
fusion/models/custom/rnn/stacked_lstm.py

Defines StackedLSTM: multiple LSTM layers stacked with dropout
applied between them.

Expected config parameters:
    input_size (int)
    hidden_size (int)
    num_layers (int): Must be >= 2 for inter_layer_dropout
        to have any effect.
    inter_layer_dropout (float): Dropout applied between
        stacked layers.
    return_all_states (bool)
"""

import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    """
    Multiple LSTM layers stacked via nn.LSTM(num_layers=...),
    with inter-layer dropout.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        inter_layer_dropout: float = 0.0,
        return_all_states: bool = False,
    ):
        """
        Args:
            input_size: Number of expected features in the input.
            hidden_size: Hidden size for every stacked layer.
            num_layers: Number of stacked LSTM layers.
            inter_layer_dropout: Dropout applied to the output of
                every layer except the last (nn.LSTM's native behavior).
            return_all_states: Output mode flag (see forward()).
        """
        super().__init__()

        self.return_all_states = return_all_states

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=inter_layer_dropout if num_layers > 1 else 0.0,
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
        outputs, (h_n, _c_n) = self.lstm(x)

        if self.return_all_states:
            return outputs

        return h_n[-1]

    @classmethod
    def from_config(cls, config) -> "StackedLSTM":
        """
        Build a StackedLSTM from a Config object.
        """
        return cls(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.get("num_layers", 2),
            inter_layer_dropout=config.get(
                "inter_layer_dropout",
                0.0,
            ),
            return_all_states=config.get(
                "return_all_states",
                False,
            ),
        )