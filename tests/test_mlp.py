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


# ============================================================
# MixtureOfExperts Unit Tests
# ============================================================

import pytest
import torch

from fusion.models.custom.mlp.mixture_of_experts import MixtureOfExperts


def test_moe_output_shape():
    model = MixtureOfExperts(
        input_dim=10,
        hidden_dims=[16],
        output_dim=4,
        num_experts=6,
        top_k=2,
    )

    out = model(torch.randn(8, 10))

    assert out.shape == (8, 4)


def test_moe_aux_loss_is_populated_and_finite():
    model = MixtureOfExperts(
        input_dim=6,
        hidden_dims=[8],
        output_dim=2,
        num_experts=4,
        top_k=2,
    )

    model(torch.randn(5, 6))

    assert torch.isfinite(model.aux_loss)
    assert model.aux_loss.item() >= 0


def test_moe_top_k_equal_num_experts_uses_all():
    model = MixtureOfExperts(
        input_dim=4,
        hidden_dims=[8],
        output_dim=2,
        num_experts=3,
        top_k=3,
    )

    out = model(torch.randn(3, 4))

    assert out.shape == (3, 2)


def test_moe_invalid_top_k_raises():
    with pytest.raises(AssertionError):
        MixtureOfExperts(
            input_dim=4,
            hidden_dims=[8],
            output_dim=2,
            num_experts=2,
            top_k=5,
        )


def test_moe_aux_loss_penalizes_imbalance_more_than_balance():
    torch.manual_seed(0)

    model = MixtureOfExperts(
        input_dim=4,
        hidden_dims=[8],
        output_dim=2,
        num_experts=4,
        top_k=2,
    )

    balanced = torch.full((4, 4), 0.25)

    imbalanced = torch.tensor(
        [
            [0.9, 0.05, 0.03, 0.02],
        ] * 4
    )

    loss_balanced = model._load_balancing_loss(balanced)
    loss_imbalanced = model._load_balancing_loss(imbalanced)

    assert loss_imbalanced > loss_balanced


# ============================================================
# AdaptiveMLP Unit Tests
# ============================================================

import torch
import torch.nn as nn

from fusion.models.custom.mlp.adaptive_mlp import AdaptiveMLP


def test_adaptive_mlp_training_mode_returns_one_tuple_per_layer():
    model = AdaptiveMLP(
        input_dim=10,
        hidden_dim=16,
        num_layers=4,
        output_dim=3,
    )

    model.train()

    outputs = model(torch.randn(5, 10))

    assert len(outputs) == 4

    for prediction, confidence in outputs:
        assert prediction.shape == (5, 3)
        assert confidence.shape == (5, 1)
        assert torch.all(confidence >= 0)
        assert torch.all(confidence <= 1)


def test_adaptive_mlp_eval_mode_returns_single_tensor():
    model = AdaptiveMLP(
        input_dim=8,
        hidden_dim=16,
        num_layers=3,
        output_dim=2,
    )

    model.eval()

    out = model(torch.randn(4, 8))

    assert out.shape == (4, 2)


def test_adaptive_mlp_low_threshold_exits_on_first_layer():
    model = AdaptiveMLP(
        input_dim=6,
        hidden_dim=8,
        num_layers=5,
        output_dim=2,
        exit_threshold=-1.0,
    )

    model.eval()

    out = model(torch.randn(3, 6))

    assert out.shape == (3, 2)


def test_adaptive_mlp_compute_loss_sums_per_exit_losses():
    model = AdaptiveMLP(
        input_dim=5,
        hidden_dim=8,
        num_layers=3,
        output_dim=2,
    )

    model.train()

    x = torch.randn(4, 5)
    target = torch.randint(0, 2, (4,))

    outputs = model(x)

    loss = AdaptiveMLP.compute_loss(
        outputs,
        target,
        nn.CrossEntropyLoss(),
    )

    assert loss.dim() == 0
    assert torch.isfinite(loss)

