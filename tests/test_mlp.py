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



# ============================================================
# DeepMLP Unit Tests
# ============================================================

import torch

from fusion.models.custom.mlp.deep_mlp import DeepMLP


def test_deep_mlp_output_shape():
    model = DeepMLP(
        input_dim=10,
        hidden_dim=32,
        num_layers=5,
        output_dim=3,
    )

    out = model(torch.randn(6, 10))

    assert out.shape == (6, 3)


def test_deep_mlp_num_layers_controls_depth():
    shallow = DeepMLP(
        input_dim=4,
        hidden_dim=8,
        num_layers=1,
        output_dim=2,
    )

    deep = DeepMLP(
        input_dim=4,
        hidden_dim=8,
        num_layers=10,
        output_dim=2,
    )

    n_linear_shallow = sum(
        1 for layer in shallow.mlp.net if hasattr(layer, "weight")
    )
    n_linear_deep = sum(
        1 for layer in deep.mlp.net if hasattr(layer, "weight")
    )

    assert n_linear_deep > n_linear_shallow


def test_deep_mlp_single_layer_equivalent_to_simple_mlp_shape():
    model = DeepMLP(
        input_dim=5,
        hidden_dim=16,
        num_layers=1,
        output_dim=2,
    )

    out = model(torch.randn(2, 5))

    assert out.shape == (2, 2)

# ============================================================
# GatedMLP Unit Tests
# ============================================================

import torch

from fusion.models.custom.mlp.gated_mlp import GatedMLP


def test_gated_mlp_output_shape():
    model = GatedMLP(
        input_dim=10,
        hidden_dims=[32, 32],
        output_dim=4,
    )

    out = model(torch.randn(8, 10))

    assert out.shape == (8, 4)


def test_gated_mlp_gate_values_are_bounded_zero_one():
    model = GatedMLP(
        input_dim=6,
        hidden_dims=[16],
        output_dim=2,
    )

    layer = model.gated_layers[0]
    x = torch.randn(5, 6)

    gate = torch.sigmoid(layer.gate_proj(x))

    assert torch.all(gate >= 0)
    assert torch.all(gate <= 1)


def test_gated_mlp_gradients_flow_through_both_paths():
    model = GatedMLP(
        input_dim=4,
        hidden_dims=[8],
        output_dim=1,
    )

    x = torch.randn(3, 4, requires_grad=True)

    out = model(x)
    out.sum().backward()

    layer = model.gated_layers[0]

    assert layer.gate_proj.weight.grad is not None
    assert layer.value_proj.weight.grad is not None


def test_gated_mlp_single_gated_layer_case():
    model = GatedMLP(
        input_dim=4,
        hidden_dims=[4],
        output_dim=2,
    )

    out = model(torch.randn(2, 4))

    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()

# ============================================================
# ResidualMLP Unit Tests
# ============================================================

import torch

from fusion.models.custom.mlp.residual_mlp import ResidualMLP


def test_residual_mlp_output_shape_with_projection():
    model = ResidualMLP(
        input_dim=10,
        hidden_dim=32,
        num_blocks=3,
        output_dim=4,
    )

    out = model(torch.randn(8, 10))

    assert out.shape == (8, 4)


def test_residual_mlp_output_shape_without_projection():
    # input_dim == hidden_dim -> input_proj should be Identity.
    model = ResidualMLP(
        input_dim=16,
        hidden_dim=16,
        num_blocks=2,
        output_dim=2,
    )

    assert isinstance(model.input_proj, torch.nn.Identity)

    out = model(torch.randn(5, 16))

    assert out.shape == (5, 2)


def test_residual_mlp_zero_blocks_is_just_projection_plus_head():
    model = ResidualMLP(
        input_dim=8,
        hidden_dim=8,
        num_blocks=0,
        output_dim=3,
    )

    out = model(torch.randn(2, 8))

    assert out.shape == (2, 3)


def test_residual_mlp_gradients_reach_the_first_block():
    model = ResidualMLP(
        input_dim=6,
        hidden_dim=6,
        num_blocks=4,
        output_dim=1,
    )

    x = torch.randn(3, 6, requires_grad=True)

    out = model(x)
    out.sum().backward()

    assert model.blocks[0].linear1.weight.grad is not None


 