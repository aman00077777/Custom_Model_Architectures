"""
fusion/models/custom/rnn/attention_lstm.py

Defines AttentionLSTM: an LSTM whose time-step outputs are combined
via learned (additive) attention into a single context vector.

Expected config parameters:
    input_size (int)
    hidden_size (int)
    num_layers (int)
    dropout (float): Dropout between LSTM layers
        (ignored if num_layers == 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    """
    LSTM with learned attention pooling over all time steps.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_size: Number of expected features in the input.
            hidden_size: Hidden size of the underlying LSTM.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between layers.
                nn.LSTM ignores this if num_layers == 1.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention_scorer = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, input_size).

        Returns:
            Context tensor of shape (B, hidden_size),
            the attention-weighted summary of all time steps.
        """
        context, _weights = self.forward_with_weights(x)

        return context

    def forward_with_weights(self, x: torch.Tensor):
        """
        Same as forward(), but also returns the attention weights.

        Args:
            x: Tensor of shape (B, T, input_size).

        Returns:
            context:
                Tensor of shape (B, hidden_size).

            weights:
                Tensor of shape (B, T, 1) whose values sum
                to 1 across the sequence dimension.
        """
        outputs, _ = self.lstm(x)

        scores = self.attention_scorer(outputs)

        weights = F.softmax(
            scores,
            dim=1,
        )

        context = (
            weights * outputs
        ).sum(dim=1)

        return context, weights

    @classmethod
    def from_config(cls, config) -> "AttentionLSTM":
        """
        Build an AttentionLSTM from a Config object.
        """
        return cls(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.0),
        )