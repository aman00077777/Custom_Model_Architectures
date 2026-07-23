"""
fusion/models/custom/rnn/bidirectional_lstm.py

Defines BidirectionalLSTM: an LSTM that reads the sequence in both
directions and concatenates the final forward and backward hidden states.

Expected config parameters:
    input_size (int)
    hidden_size (int)
    num_layers (int)
    dropout (float): Dropout between LSTM layers
        (ignored if num_layers == 1).
"""

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM whose output is the concatenation of
    the final forward and backward hidden states.
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
            hidden_size: Hidden size per direction.
                Output feature dimension is 2 * hidden_size.
            num_layers: Number of stacked bidirectional LSTM layers.
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
            bidirectional=True,
        )

        self.output_dim = hidden_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, input_size).

        Returns:
            Tensor of shape (B, 2 * hidden_size) containing the
            concatenated final forward and backward hidden states.
        """
        _outputs, (h_n, _c_n) = self.lstm(x)

        forward_final = h_n[-2]
        backward_final = h_n[-1]

        return torch.cat(
            [forward_final, backward_final],
            dim=-1,
        )

    @classmethod
    def from_config(cls, config) -> "BidirectionalLSTM":
        """
        Build a BidirectionalLSTM from a Config object.
        """
        return cls( 
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.get("num_layers", 1),
            dropout=config.get("dropout", 0.0),
        )