"""
fusion/models/custom/rnn/encoder_decoder_rnn.py

Defines EncoderDecoderRNN: a Bidirectional LSTM encoder with an
attention-driven LSTMCell decoder using Bahdanau (additive) attention
for sequence-to-sequence tasks.

Expected config parameters:
    input_size (int): Encoder input feature size.
    encoder_hidden_size (int): Hidden size per direction of the encoder.
    decoder_input_size (int): Feature size of each decoder input step.
    decoder_hidden_size (int): Hidden size of the decoder LSTMCell.
    output_size (int): Feature size of each decoder prediction.
    max_target_len (int): Number of decoding steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.
    """

    def __init__(
        self,
        decoder_hidden_size: int,
        encoder_output_size: int,
        attn_dim: int = 64,
    ):
        """
        Args:
            decoder_hidden_size: Size of the decoder hidden state.
            encoder_output_size: Size of encoder outputs.
            attn_dim: Internal attention dimension.
        """
        super().__init__()

        self.decoder_proj = nn.Linear(
            decoder_hidden_size,
            attn_dim,
        )

        self.encoder_proj = nn.Linear(
            encoder_output_size,
            attn_dim,
        )

        self.scorer = nn.Linear(
            attn_dim,
            1,
        )

    def forward(
        self,
        decoder_h: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ):
        """
        Args:
            decoder_h:
                Tensor of shape (B, decoder_hidden_size).

            encoder_outputs:
                Tensor of shape
                (B, T_src, encoder_output_size).

        Returns:
            context:
                Tensor of shape
                (B, encoder_output_size).

            weights:
                Tensor of shape
                (B, T_src, 1).
        """
        decoder_term = self.decoder_proj(
            decoder_h
        ).unsqueeze(1)

        encoder_term = self.encoder_proj(
            encoder_outputs
        )

        scores = self.scorer(
            torch.tanh(
                decoder_term + encoder_term
            )
        )

        weights = F.softmax(
            scores,
            dim=1,
        )

        context = (
            weights * encoder_outputs
        ).sum(dim=1)

        return context, weights


class EncoderDecoderRNN(nn.Module):
    """
    Bidirectional LSTM encoder with Bahdanau attention and
    an LSTMCell decoder.
    """

    def __init__(
        self,
        input_size: int,
        encoder_hidden_size: int,
        decoder_input_size: int,
        decoder_hidden_size: int,
        output_size: int,
        max_target_len: int,
    ):
        """
        Args:
            input_size: Encoder input feature size.
            encoder_hidden_size: Hidden size per direction
                of the bidirectional encoder.
            decoder_input_size: Decoder input feature size.
            decoder_hidden_size: Hidden size of the decoder.
            output_size: Decoder output feature size.
            max_target_len: Number of decoding steps.
        """
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        encoder_output_size = (
            encoder_hidden_size * 2
        )

        self.attention = _BahdanauAttention(
            decoder_hidden_size,
            encoder_output_size,
        )

        self.decoder_cell = nn.LSTMCell(
            input_size=(
                decoder_input_size
                + encoder_output_size
            ),
            hidden_size=decoder_hidden_size,
        )

        self.output_proj = nn.Linear(
            decoder_hidden_size,
            output_size,
        )

        self.max_target_len = max_target_len
        self.decoder_hidden_size = (
            decoder_hidden_size
        )

    def forward(
        self,
        source: torch.Tensor,
        decoder_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            source:
                Tensor of shape
                (B, T_src, input_size).

            decoder_inputs:
                Tensor of shape
                (B, max_target_len, decoder_input_size).

        Returns:
            Tensor of shape
            (B, max_target_len, output_size).
        """
        batch_size = source.shape[0]
        device = source.device

        encoder_outputs, _ = self.encoder(
            source
        )

        h = torch.zeros(
            batch_size,
            self.decoder_hidden_size,
            device=device,
        )

        c = torch.zeros(
            batch_size,
            self.decoder_hidden_size,
            device=device,
        )

        predictions = []

        for t in range(self.max_target_len):
            context, _weights = self.attention(
                h,
                encoder_outputs,
            )

            step_input = torch.cat(
                [
                    decoder_inputs[:, t, :],
                    context,
                ],
                dim=-1,
            )

            h, c = self.decoder_cell(
                step_input,
                (h, c),
            )

            predictions.append(
                self.output_proj(h)
            )

        return torch.stack(
            predictions,
            dim=1,
        )

    @classmethod
    def from_config(
        cls,
        config,
    ) -> "EncoderDecoderRNN":
        """
        Build an EncoderDecoderRNN from a Config object.
        """
        return cls(
            input_size=config.input_size,
            encoder_hidden_size=config.encoder_hidden_size,
            decoder_input_size=config.decoder_input_size,
            decoder_hidden_size=config.decoder_hidden_size,
            output_size=config.output_size,
            max_target_len=config.max_target_len,
        )