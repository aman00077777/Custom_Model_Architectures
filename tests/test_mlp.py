# ============================================================
# SimpleMLP Tests
# ============================================================

import torch
import torch.nn as nn

from fusion.models.custom.mlp.simple_mlp import SimpleMLP


def test_simple_mlp_output_shape():
    model = SimpleMLP(
        input_dim=10,
        hidden_dims=[32, 16],
        output_dim=4,
    )

    out = model(torch.randn(8, 10))

    assert out.shape == (8, 4)


def test_simple_mlp_empty_hidden_dims_is_single_linear():
    model = SimpleMLP(
        input_dim=5,
        hidden_dims=[],
        output_dim=2,
    )

    assert sum(isinstance(m, nn.Linear) for m in model.net) == 1

    out = model(torch.randn(3, 5))

    assert out.shape == (3, 2)


def test_simple_mlp_batchnorm_and_dropout_dont_break_forward():
    model = SimpleMLP(
        input_dim=6,
        hidden_dims=[12, 12],
        output_dim=3,
        use_batchnorm=True,
        dropout_rate=0.5,
    )

    out = model(torch.randn(4, 6))

    assert out.shape == (4, 3)
    assert torch.isfinite(out).all()


def test_simple_mlp_no_activation_after_final_layer():
    model = SimpleMLP(
        input_dim=4,
        hidden_dims=[8],
        output_dim=2,
    )

    last_module = model.net[-1]

    assert isinstance(last_module, nn.Linear)


def test_simple_mlp_from_config():
    class Cfg:
        input_dim = 8
        hidden_dims = [16]
        output_dim = 2

        def get(self, k, d=None):
            return getattr(self, k, d)

    model = SimpleMLP.from_config(Cfg())

    out = model(torch.randn(2, 8))

    assert out.shape == (2, 2)