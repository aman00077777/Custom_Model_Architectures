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


