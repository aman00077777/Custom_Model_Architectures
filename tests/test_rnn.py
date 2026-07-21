# ============================================================
# LSTMEncoder Unit Tests
# ============================================================

import torch

from fusion.models.custom.rnn.lstm import LSTMEncoder


def test_lstm_last_state_shape():
    model = LSTMEncoder(
        input_size=8,
        hidden_size=16,
        num_layers=2,
    )

    out = model(torch.randn(4, 10, 8))

    assert out.shape == (4, 16)


def test_lstm_all_states_shape():
    model = LSTMEncoder(
        input_size=8,
        hidden_size=16,
        return_all_states=True,
    )

    out = model(torch.randn(4, 10, 8))

    assert out.shape == (4, 10, 16)


def test_lstm_single_layer_ignores_dropout_without_erroring():
    model = LSTMEncoder(
        input_size=4,
        hidden_size=8,
        num_layers=1,
        dropout=0.5,
    )

    out = model(torch.randn(2, 5, 4))

    assert out.shape == (2, 8)


def test_lstm_variable_sequence_lengths():
    model = LSTMEncoder(
        input_size=4,
        hidden_size=6,
        return_all_states=True,
    )

    short = model(torch.randn(2, 3, 4))
    long = model(torch.randn(2, 20, 4))

    assert short.shape == (2, 3, 6)
    assert long.shape == (2, 20, 6)


# ============================================================
# GRUEncoder Unit Tests
# ============================================================

import torch

from fusion.models.custom.rnn.gru import GRUEncoder


def test_gru_last_state_shape():
    model = GRUEncoder(
        input_size=8,
        hidden_size=16,
        num_layers=2,
    )

    out = model(torch.randn(4, 10, 8))

    assert out.shape == (4, 16)


def test_gru_all_states_shape():
    model = GRUEncoder(
        input_size=8,
        hidden_size=16,
        return_all_states=True,
    )

    out = model(torch.randn(4, 10, 8))

    assert out.shape == (4, 10, 16)


def test_gru_has_fewer_params_than_equivalent_lstm():
    from fusion.models.custom.rnn.lstm import LSTMEncoder

    gru = GRUEncoder(
        input_size=8,
        hidden_size=16,
    )

    lstm = LSTMEncoder(
        input_size=8,
        hidden_size=16,
    )

    gru_params = sum(p.numel() for p in gru.parameters())
    lstm_params = sum(p.numel() for p in lstm.parameters())

    assert gru_params < lstm_params

# ============================================================
# BidirectionalLSTM Unit Tests
# ============================================================

import torch

from fusion.models.custom.rnn.bidirectional_lstm import BidirectionalLSTM


def test_bilstm_output_dim_is_double_hidden_size():
    model = BidirectionalLSTM(
        input_size=8,
        hidden_size=16,
    )

    out = model(torch.randn(4, 10, 8))

    assert out.shape == (4, 32)
    assert model.output_dim == 32


def test_bilstm_multi_layer_bidirectional():
    model = BidirectionalLSTM(
        input_size=6,
        hidden_size=10,
        num_layers=3,
        dropout=0.2,
    )

    out = model(torch.randn(3, 7, 6))

    assert out.shape == (3, 20)


def test_bilstm_forward_and_backward_states_differ():
    torch.manual_seed(0)

    model = BidirectionalLSTM(
        input_size=4,
        hidden_size=8,
    )

    _outputs, (h_n, _c_n) = model.lstm(torch.randn(2, 5, 4))

    assert not torch.allclose(
        h_n[-2],
        h_n[-1],
    )

# ============================================================
# StackedLSTM Unit Tests
# ============================================================

import torch

from fusion.models.custom.rnn.stacked_lstm import StackedLSTM


def test_stacked_lstm_output_shape_last_state():
    model = StackedLSTM(
        input_size=8,
        hidden_size=16,
        num_layers=4,
    )

    out = model(torch.randn(4, 10, 8))

    assert out.shape == (4, 16)


def test_stacked_lstm_output_shape_all_states():
    model = StackedLSTM(
        input_size=8,
        hidden_size=16,
        num_layers=3,
        return_all_states=True,
    )

    out = model(torch.randn(4, 10, 8))

    assert out.shape == (4, 10, 16)


def test_stacked_lstm_dropout_only_applies_with_multiple_layers():
    # num_layers=1 should silently zero out dropout instead of
    # raising a UserWarning or error.
    model = StackedLSTM(
        input_size=4,
        hidden_size=8,
        num_layers=1,
        inter_layer_dropout=0.5,
    )

    out = model(torch.randn(2, 5, 4))

    assert out.shape == (2, 8)


def test_stacked_lstm_more_layers_means_more_parameters():
    shallow = StackedLSTM(
        input_size=4,
        hidden_size=8,
        num_layers=1,
    )

    deep = StackedLSTM(
        input_size=4,
        hidden_size=8,
        num_layers=5,
    )

    shallow_params = sum(
        p.numel() for p in shallow.parameters()
    )
    deep_params = sum(
        p.numel() for p in deep.parameters()
    )

    assert deep_params > shallow_params

# ============================================================
# AttentionLSTM Unit Tests
# ============================================================

import torch

from fusion.models.custom.rnn.attention_lstm import AttentionLSTM


def test_attn_lstm_context_shape():
    model = AttentionLSTM(
        input_size=8,
        hidden_size=16,
    )

    out = model(torch.randn(4, 12, 8))

    assert out.shape == (4, 16)


def test_attn_lstm_attention_weights_sum_to_one_over_time():
    model = AttentionLSTM(
        input_size=6,
        hidden_size=10,
    )

    _context, weights = model.forward_with_weights(
        torch.randn(3, 9, 6)
    )

    assert weights.shape == (3, 9, 1)

    sums = weights.sum(dim=1).squeeze(-1)

    assert torch.allclose(
        sums,
        torch.ones(3),
        atol=1e-5,
    )


def test_attn_lstm_weights_are_nonnegative():
    model = AttentionLSTM(
        input_size=4,
        hidden_size=8,
    )

    _context, weights = model.forward_with_weights(
        torch.randn(2, 5, 4)
    )

    assert torch.all(weights >= 0)


def test_attn_lstm_gradients_flow_to_attention_scorer():
    model = AttentionLSTM(
        input_size=4,
        hidden_size=8,
    )

    x = torch.randn(2, 5, 4)

    out = model(x)

    out.sum().backward()

    assert model.attention_scorer.weight.grad is not None

# ============================================================
# EncoderDecoderRNN Unit Tests
# ============================================================

import torch

from fusion.models.custom.rnn.encoder_decoder_rnn import (
    EncoderDecoderRNN,
)


def _make_model():
    return EncoderDecoderRNN(
        input_size=8,
        encoder_hidden_size=16,
        decoder_input_size=5,
        decoder_hidden_size=12,
        output_size=6,
        max_target_len=7,
    )


def test_enc_dec_rnn_output_shape():
    model = _make_model()

    source = torch.randn(3, 10, 8)
    decoder_inputs = torch.randn(3, 7, 5)

    out = model(
        source,
        decoder_inputs,
    )

    assert out.shape == (3, 7, 6)


def test_enc_dec_rnn_different_source_lengths_still_work():
    model = _make_model()

    decoder_inputs = torch.randn(2, 7, 5)

    out_short = model(
        torch.randn(2, 4, 8),
        decoder_inputs,
    )

    out_long = model(
        torch.randn(2, 25, 8),
        decoder_inputs,
    )

    assert out_short.shape == (2, 7, 6)
    assert out_long.shape == (2, 7, 6)


def test_enc_dec_rnn_attention_weights_sum_to_one():
    model = _make_model()

    encoder_outputs = torch.randn(
        3,
        10,
        32,
    )  # 2 * encoder_hidden_size

    decoder_h = torch.randn(3, 12)

    _context, weights = model.attention(
        decoder_h,
        encoder_outputs,
    )

    sums = weights.sum(dim=1).squeeze(-1)

    assert torch.allclose(
        sums,
        torch.ones(3),
        atol=1e-5,
    )


def test_enc_dec_rnn_gradients_flow_back_to_encoder():
    model = _make_model()

    source = torch.randn(
        2,
        6,
        8,
        requires_grad=True,
    )

    decoder_inputs = torch.randn(2, 7, 5)

    out = model(
        source,
        decoder_inputs,
    )

    out.sum().backward()

    assert source.grad is not None
    assert torch.isfinite(source.grad).all()





