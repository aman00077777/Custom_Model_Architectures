# pytest tests/test_cnn.py -x -v
"""
Strict, exhaustive test suite for all CNN building blocks in
``fusion.models.custom.cnn``.

Execution contract:
    -x  → halt instantly on the first failure.
    -v  → verbose per-test reporting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

# ── Module imports ──────────────────────────────────────────────
from fusion.models.custom.cnn.simple_cnn import SimpleCNN
from fusion.models.custom.cnn.residual_cnn import ResidualCNN, ResidualBlock
from fusion.models.custom.cnn.multi_scale_cnn import MultiScaleCNN, MultiScaleBlock
from fusion.models.custom.cnn.lightweight_cnn import LightweightCNN
from fusion.models.custom.cnn.inception_cnn import InceptionCNN, InceptionModule
from fusion.models.custom.cnn.dense_cnn import DenseCNN
from fusion.models.custom.cnn.custom_cnn import CustomCNN

# ================================================================
# Helper configs
# ================================================================

BATCH = 2
IN_CH = 3
H, W = 64, 64
DUMMY_INPUT = torch.randn(BATCH, IN_CH, H, W)


def _simple_cfg() -> dict:
    return {
        "in_channels": IN_CH,
        "out_channels": [16, 32, 64],
        "kernel_sizes": [3, 3, 3],
        "strides": [1, 1, 1],
        "use_batchnorm": True,
        "dropout_rate": 0.0,
        "activation": "relu",
    }


def _residual_cfg() -> dict:
    return {
        "in_channels": IN_CH,
        "block_channels": [16, 32, 64],
        "kernel_size": 3,
        "activation": "relu",
    }


def _multi_scale_cfg() -> dict:
    return {
        "in_channels": IN_CH,
        "branch_kernel_sizes": [3, 5, 7],
        "branch_out_channels": 16,
        "output_channels": 64,
        "num_blocks": 1,
        "activation": "relu",
    }


def _lightweight_cfg() -> dict:
    return {
        "in_channels": IN_CH,
        "block_channels": [16, 32, 64],
        "kernel_size": 3,
        "activation": "relu",
    }


def _inception_cfg() -> dict:
    return {
        "in_channels": IN_CH,
        "num_blocks": 2,
        "ch_1x1": 16,
        "ch_reduce_3x3": 16,
        "ch_3x3": 32,
        "ch_reduce_5x5": 8,
        "ch_5x5": 16,
        "ch_pool_proj": 16,
        "activation": "relu",
    }


def _dense_cfg() -> dict:
    return {
        "in_channels": IN_CH,
        "growth_rate": 12,
        "num_layers_per_block": [4, 4],
        "compression": 0.5,
        "activation": "relu",
    }


def _custom_cfg() -> dict:
    return {
        "in_channels": IN_CH,
        "layers": [
            {"type": "conv", "out_channels": 32, "kernel_size": 3, "stride": 1},
            {"type": "batchnorm"},
            {"type": "activation", "name": "relu"},
            {"type": "conv", "out_channels": 64, "kernel_size": 3, "stride": 1},
            {"type": "batchnorm"},
            {"type": "activation", "name": "relu"},
        ],
    }


# ================================================================
# 1. OUTPUT DIMENSIONALITY TEST — all 7 models
# ================================================================

class TestOutputDimensionality:
    """For every CNN, a (B, C, H, W) input must yield exactly (B, feature_dim)
    with spatial dimensions fully collapsed."""

    def test_simple_cnn_output_shape(self) -> None:
        model = SimpleCNN(_simple_cfg())
        out = model(DUMMY_INPUT)
        assert out.shape == (BATCH, model.output_dim)
        assert out.ndim == 2, "Spatial dims must be fully collapsed."

    def test_residual_cnn_output_shape(self) -> None:
        model = ResidualCNN(_residual_cfg())
        out = model(DUMMY_INPUT)
        assert out.shape == (BATCH, model.output_dim)
        assert out.ndim == 2

    def test_multi_scale_cnn_output_shape(self) -> None:
        model = MultiScaleCNN(_multi_scale_cfg())
        out = model(DUMMY_INPUT)
        assert out.shape == (BATCH, model.output_dim)
        assert out.ndim == 2

    def test_lightweight_cnn_output_shape(self) -> None:
        model = LightweightCNN(_lightweight_cfg())
        out = model(DUMMY_INPUT)
        assert out.shape == (BATCH, model.output_dim)
        assert out.ndim == 2

    def test_inception_cnn_output_shape(self) -> None:
        model = InceptionCNN(_inception_cfg())
        out = model(DUMMY_INPUT)
        assert out.shape == (BATCH, model.output_dim)
        assert out.ndim == 2

    def test_dense_cnn_output_shape(self) -> None:
        model = DenseCNN(_dense_cfg())
        out = model(DUMMY_INPUT)
        assert out.shape == (BATCH, model.output_dim)
        assert out.ndim == 2

    def test_custom_cnn_output_shape(self) -> None:
        model = CustomCNN(_custom_cfg())
        out = model(DUMMY_INPUT)
        assert out.shape == (BATCH, model.output_dim)
        assert out.ndim == 2


# ================================================================
# 2. RESOLUTION INDEPENDENCE TEST — AdaptiveAvgPool2d correctness
# ================================================================

class TestResolutionIndependence:
    """Varying spatial input sizes must produce an identical feature_dim
    thanks to AdaptiveAvgPool2d."""

    @pytest.mark.parametrize("spatial", [32, 64, 128, 224])
    def test_simple_cnn_resolution_invariance(self, spatial: int) -> None:
        cfg = _simple_cfg()
        model = SimpleCNN(cfg)
        x = torch.randn(BATCH, IN_CH, spatial, spatial)
        out = model(x)
        assert out.shape == (BATCH, model.output_dim), (
            f"feature_dim must stay {model.output_dim} at resolution {spatial}×{spatial}"
        )

    @pytest.mark.parametrize("spatial", [32, 64, 128, 224])
    def test_residual_cnn_resolution_invariance(self, spatial: int) -> None:
        cfg = _residual_cfg()
        model = ResidualCNN(cfg)
        x = torch.randn(BATCH, IN_CH, spatial, spatial)
        out = model(x)
        assert out.shape == (BATCH, model.output_dim), (
            f"feature_dim must stay {model.output_dim} at resolution {spatial}×{spatial}"
        )

    def test_simple_vs_residual_same_feature_dim_across_sizes(self) -> None:
        """Two different resolutions must agree on feature_dim for the
        same model instance."""
        for ModelCls, cfg_fn in [
            (SimpleCNN, _simple_cfg),
            (ResidualCNN, _residual_cfg),
        ]:
            model = ModelCls(cfg_fn())
            out_32 = model(torch.randn(BATCH, IN_CH, 32, 32))
            out_224 = model(torch.randn(BATCH, IN_CH, 224, 224))
            assert out_32.shape == out_224.shape, (
                f"{ModelCls.__name__}: shape differs between 32×32 and 224×224"
            )


# ================================================================
# 3. RESIDUAL SKIP CONNECTION TEST — channel mismatch handling
# ================================================================

class TestResidualSkipConnection:
    """When in_channels ≠ out_channels the block must apply a 1×1
    projection conv on the skip path without errors."""

    def test_block_with_channel_mismatch(self) -> None:
        block = ResidualBlock(in_channels=32, out_channels=64, kernel_size=3)
        x = torch.randn(BATCH, 32, 16, 16)
        out = block(x)
        assert out.shape == (BATCH, 64, 16, 16)

    def test_block_with_matching_channels(self) -> None:
        block = ResidualBlock(in_channels=32, out_channels=32, kernel_size=3)
        x = torch.randn(BATCH, 32, 16, 16)
        out = block(x)
        assert out.shape == (BATCH, 32, 16, 16)

    def test_skip_is_projection_when_channels_differ(self) -> None:
        block = ResidualBlock(in_channels=32, out_channels=64)
        # skip should be nn.Sequential (1×1 conv + BN), not Identity
        assert not isinstance(block.skip, nn.Identity), (
            "skip must be a learned projection when channels differ."
        )

    def test_skip_is_identity_when_channels_match(self) -> None:
        block = ResidualBlock(in_channels=32, out_channels=32)
        assert isinstance(block.skip, nn.Identity)

    def test_gradients_flow_through_skip(self) -> None:
        """Ensure the skip path is differentiable and gradients propagate."""
        block = ResidualBlock(in_channels=16, out_channels=32)
        x = torch.randn(1, 16, 8, 8, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradients must flow through skip path."
        assert x.grad.abs().sum() > 0


# ================================================================
# 4. MULTI-BRANCH CONCATENATION TEST
# ================================================================

class TestMultiBranchConcatenation:
    """Verify that channel concatenation and 1×1 fusion produce the
    correct intermediate and final dimensions."""

    # ── MultiScaleCNN ──────────────────────────────────────────

    def test_multi_scale_block_concat_channels(self) -> None:
        block = MultiScaleBlock(
            in_channels=IN_CH,
            branch_kernel_sizes=[3, 5, 7],
            branch_out_channels=16,
            output_channels=64,
        )
        # Expected concat width = 16 * 3 branches = 48
        assert block.concat_channels == 48

    def test_multi_scale_block_output_channels(self) -> None:
        block = MultiScaleBlock(
            in_channels=IN_CH,
            branch_kernel_sizes=[3, 5, 7],
            branch_out_channels=16,
            output_channels=64,
        )
        x = torch.randn(BATCH, IN_CH, 32, 32)
        out = block(x)
        # After 1×1 fusion the channel dim must be exactly output_channels
        assert out.shape[1] == 64

    def test_multi_scale_pre_fusion_channels(self) -> None:
        """Intercept the intermediate concat tensor to verify branch widths."""
        block = MultiScaleBlock(
            in_channels=IN_CH,
            branch_kernel_sizes=[3, 5],
            branch_out_channels=20,
            output_channels=64,
        )
        x = torch.randn(BATCH, IN_CH, 16, 16)
        branch_outputs = [branch(x) for branch in block.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        # 2 branches × 20 = 40
        assert concatenated.shape[1] == 40

    # ── InceptionCNN ──────────────────────────────────────────

    def test_inception_module_concat_channels(self) -> None:
        mod = InceptionModule(
            in_channels=IN_CH,
            ch_1x1=16,
            ch_reduce_3x3=8,
            ch_3x3=32,
            ch_reduce_5x5=4,
            ch_5x5=16,
            ch_pool_proj=16,
        )
        # concat = 16 + 32 + 16 + 16 = 80
        assert mod.out_channels == 80

    def test_inception_module_forward_shape(self) -> None:
        mod = InceptionModule(
            in_channels=IN_CH,
            ch_1x1=16,
            ch_reduce_3x3=8,
            ch_3x3=32,
            ch_reduce_5x5=4,
            ch_5x5=16,
            ch_pool_proj=16,
        )
        x = torch.randn(BATCH, IN_CH, 32, 32)
        out = mod(x)
        assert out.shape == (BATCH, 80, 32, 32)

    def test_inception_per_branch_channels(self) -> None:
        """Run each branch independently and verify individual widths."""
        mod = InceptionModule(
            in_channels=IN_CH,
            ch_1x1=16,
            ch_reduce_3x3=8,
            ch_3x3=32,
            ch_reduce_5x5=4,
            ch_5x5=16,
            ch_pool_proj=16,
        )
        x = torch.randn(BATCH, IN_CH, 16, 16)

        assert mod.branch1(x).shape[1] == 16
        assert mod.branch2(x).shape[1] == 32
        assert mod.branch3(x).shape[1] == 16
        assert mod.branch4(x).shape[1] == 16


# ================================================================
# 5. DYNAMIC CONFIG TEST — CustomCNN
# ================================================================

class TestDynamicConfig:
    """The CustomCNN's internal Sequential must mirror the config spec."""

    def test_num_config_layers_matches(self) -> None:
        cfg = _custom_cfg()
        model = CustomCNN(cfg)
        assert model.num_config_layers == len(cfg["layers"])

    def test_sequential_module_count(self) -> None:
        """The nn.Sequential should contain exactly len(layers) + 2
        modules (the extra 2 being AdaptiveAvgPool2d and Flatten)."""
        cfg = _custom_cfg()
        model = CustomCNN(cfg)
        expected_total = len(cfg["layers"]) + 2  # pool + flatten
        assert len(model.net) == expected_total

    def test_varying_config_lengths(self) -> None:
        """Try multiple config lengths and verify module count."""
        for n_conv in [1, 3, 5]:
            layers = []
            for _ in range(n_conv):
                layers.append(
                    {"type": "conv", "out_channels": 16, "kernel_size": 3}
                )
                layers.append({"type": "activation", "name": "relu"})
            cfg = {"in_channels": IN_CH, "layers": layers}
            model = CustomCNN(cfg)
            assert model.num_config_layers == len(layers)
            assert len(model.net) == len(layers) + 2

    def test_unsupported_layer_raises(self) -> None:
        cfg = {
            "in_channels": IN_CH,
            "layers": [{"type": "transformer_block"}],
        }
        with pytest.raises(ValueError, match="Unsupported layer type"):
            CustomCNN(cfg)

    def test_custom_cnn_all_layer_types(self) -> None:
        """Build a model exercising every supported layer type."""
        cfg = {
            "in_channels": IN_CH,
            "layers": [
                {"type": "conv", "out_channels": 16, "kernel_size": 3},
                {"type": "batchnorm"},
                {"type": "activation", "name": "relu"},
                {"type": "dropout", "rate": 0.1},
                {"type": "maxpool", "kernel_size": 2, "stride": 2},
                {"type": "conv", "out_channels": 32, "kernel_size": 3},
                {"type": "avgpool", "kernel_size": 2, "stride": 2},
            ],
        }
        model = CustomCNN(cfg)
        out = model(torch.randn(BATCH, IN_CH, 64, 64))
        assert out.shape == (BATCH, model.output_dim)
        assert out.ndim == 2


# ================================================================
# 6. NUMERICAL SANITY — no NaN / Inf
# ================================================================

class TestNumericalSanity:
    """All models must produce finite outputs on random inputs."""

    @pytest.mark.parametrize(
        "cls,cfg_fn",
        [
            (SimpleCNN, _simple_cfg),
            (ResidualCNN, _residual_cfg),
            (MultiScaleCNN, _multi_scale_cfg),
            (LightweightCNN, _lightweight_cfg),
            (InceptionCNN, _inception_cfg),
            (DenseCNN, _dense_cfg),
            (CustomCNN, _custom_cfg),
        ],
        ids=[
            "SimpleCNN",
            "ResidualCNN",
            "MultiScaleCNN",
            "LightweightCNN",
            "InceptionCNN",
            "DenseCNN",
            "CustomCNN",
        ],
    )
    def test_output_is_finite(self, cls, cfg_fn) -> None:
        model = cls(cfg_fn())
        model.eval()
        with torch.no_grad():
            out = model(DUMMY_INPUT)
        assert torch.isfinite(out).all(), f"{cls.__name__} produced NaN/Inf."


# ================================================================
# 7. EXPORT SMOKE TEST — __all__ surface area
# ================================================================

class TestPackageExports:
    """Verify that ``fusion.models.custom.cnn`` exports all classes."""

    def test_all_classes_exported(self) -> None:
        from fusion.models.custom.cnn import __all__

        expected = {
            "BaseCNN",
            "SimpleCNN",
            "ResidualCNN",
            "MultiScaleCNN",
            "LightweightCNN",
            "InceptionCNN",
            "DenseCNN",
            "CustomCNN",
        }
        assert expected.issubset(set(__all__)), (
            f"Missing from __all__: {expected - set(__all__)}"
        )
