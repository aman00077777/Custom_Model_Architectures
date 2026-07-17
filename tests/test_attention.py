# Run with: pytest tests/ -x -v
"""
Strict TDD test suite for Attention modules.

Execute:
    pytest tests/ -x -v

The -x flag halts immediately on the first failure.
"""

from __future__ import annotations

import math

import pytest
import torch

from fusion.models.custom.attention import (
    CrossAttention,
    MultiHeadAttention,
    SelfAttention,
    SpatialAttention,
)


# ---------------------------------------------------------------------------
# SelfAttention Tests
# ---------------------------------------------------------------------------

class TestSelfAttention:
    """Exhaustive tests for SelfAttention."""

    BATCH = 2
    SEQ_LEN = 8
    EMBED_DIM = 32

    def _model(self, dropout: float = 0.0) -> SelfAttention:
        return SelfAttention(embed_dim=self.EMBED_DIM, dropout=dropout)

    def _input(self) -> torch.Tensor:
        torch.manual_seed(0)
        return torch.randn(self.BATCH, self.SEQ_LEN, self.EMBED_DIM)

    # ---- Shape tests -------------------------------------------------------

    def test_output_shape_no_mask(self) -> None:
        """Output must match input shape (B, T, embed_dim)."""
        model = self._model()
        x = self._input()
        out = model(x)
        assert out.shape == (self.BATCH, self.SEQ_LEN, self.EMBED_DIM), (
            f"Expected {(self.BATCH, self.SEQ_LEN, self.EMBED_DIM)}, got {out.shape}"
        )

    def test_output_shape_with_causal_mask(self) -> None:
        """Output shape must be unchanged when a causal mask is applied."""
        model = self._model()
        x = self._input()
        T = self.SEQ_LEN
        # Upper-triangular causal mask: True → masked out
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        out = model(x, mask=mask)
        assert out.shape == (self.BATCH, self.SEQ_LEN, self.EMBED_DIM)

    # ---- Causal masking correctness ----------------------------------------

    def test_causal_mask_changes_output(self) -> None:
        """Applying a causal mask must produce different values than no mask."""
        torch.manual_seed(1)
        model = self._model()
        model.eval()
        x = self._input()

        T = self.SEQ_LEN
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)

        with torch.no_grad():
            out_unmasked = model(x)
            out_masked = model(x, mask=mask)

        assert not torch.allclose(out_unmasked, out_masked, atol=1e-6), (
            "Masked and unmasked outputs are identical — masking has no effect."
        )

    def test_causal_mask_isolates_position_0(self) -> None:
        """With a causal mask, position 0 can only attend to itself.
        Therefore, changing all tokens EXCEPT position 0 must not affect
        the output at position 0 — proving the mask correctly blocks future
        positions from influencing earlier ones."""
        torch.manual_seed(2)
        model = self._model()
        model.eval()

        T = self.SEQ_LEN

        # Two inputs that share position-0 but differ everywhere else
        x_a = torch.randn(1, T, self.EMBED_DIM)
        x_b = x_a.clone()
        x_b[:, 1:, :] = torch.randn(1, T - 1, self.EMBED_DIM)  # alter T=1..T-1

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool), diagonal=1
        )

        with torch.no_grad():
            out_a = model(x_a, mask=causal_mask)  # (1, T, embed_dim)
            out_b = model(x_b, mask=causal_mask)  # (1, T, embed_dim)

        # Position 0 is isolated by the causal mask; the two outputs must match
        assert torch.allclose(out_a[:, 0, :], out_b[:, 0, :], atol=1e-6), (
            "Position-0 outputs differ when only later tokens change — "
            "the causal mask is not blocking future positions correctly."
        )

    def test_batch_mask_shape(self) -> None:
        """A 3-D batch mask (B, T, T) must be accepted without error."""
        model = self._model()
        x = self._input()
        T = self.SEQ_LEN
        mask = torch.zeros(self.BATCH, T, T, dtype=torch.bool)
        out = model(x, mask=mask)
        assert out.shape == (self.BATCH, self.SEQ_LEN, self.EMBED_DIM)

    # ---- Module structure --------------------------------------------------

    def test_projection_layers_exist(self) -> None:
        """W_q, W_k, W_v must be nn.Linear with correct dimensions."""
        model = self._model()
        for name in ("W_q", "W_k", "W_v"):
            layer = getattr(model, name, None)
            assert layer is not None, f"Missing projection layer: {name}"
            assert isinstance(layer, torch.nn.Linear), (
                f"{name} must be nn.Linear"
            )
            assert layer.weight.shape == (self.EMBED_DIM, self.EMBED_DIM), (
                f"{name} weight shape mismatch"
            )


# ---------------------------------------------------------------------------
# CrossAttention Tests
# ---------------------------------------------------------------------------

class TestCrossAttention:
    """Exhaustive tests for CrossAttention."""

    BATCH = 2
    QUERY_DIM = 32
    KV_DIM = 64

    def _model(self) -> CrossAttention:
        return CrossAttention(query_dim=self.QUERY_DIM, kv_dim=self.KV_DIM)

    # ---- Matching sequence lengths -----------------------------------------

    def test_output_shape_equal_seq_len(self) -> None:
        """Output length == query length when T_q == T_kv."""
        model = self._model()
        T = 10
        query = torch.randn(self.BATCH, T, self.QUERY_DIM)
        key_value = torch.randn(self.BATCH, T, self.KV_DIM)
        out = model(query, key_value)
        assert out.shape == (self.BATCH, T, self.QUERY_DIM), (
            f"Expected ({self.BATCH}, {T}, {self.QUERY_DIM}), got {out.shape}"
        )

    # ---- Mismatched sequence lengths ---------------------------------------

    def test_output_shape_query_shorter(self) -> None:
        """Output must align with the QUERY length (T_q=10, T_kv=20)."""
        model = self._model()
        T_q, T_kv = 10, 20
        query = torch.randn(self.BATCH, T_q, self.QUERY_DIM)
        key_value = torch.randn(self.BATCH, T_kv, self.KV_DIM)
        out = model(query, key_value)
        assert out.shape == (self.BATCH, T_q, self.QUERY_DIM), (
            f"Expected ({self.BATCH}, {T_q}, {self.QUERY_DIM}), got {out.shape}"
        )

    def test_output_shape_query_longer(self) -> None:
        """Output must align with the QUERY length (T_q=20, T_kv=10)."""
        model = self._model()
        T_q, T_kv = 20, 10
        query = torch.randn(self.BATCH, T_q, self.QUERY_DIM)
        key_value = torch.randn(self.BATCH, T_kv, self.KV_DIM)
        out = model(query, key_value)
        assert out.shape == (self.BATCH, T_q, self.QUERY_DIM), (
            f"Expected ({self.BATCH}, {T_q}, {self.QUERY_DIM}), got {out.shape}"
        )

    def test_output_shape_asymmetric_dims(self) -> None:
        """query_dim != kv_dim must be supported without error."""
        model = CrossAttention(query_dim=16, kv_dim=128)
        T_q, T_kv = 5, 15
        query = torch.randn(self.BATCH, T_q, 16)
        key_value = torch.randn(self.BATCH, T_kv, 128)
        out = model(query, key_value)
        assert out.shape == (self.BATCH, T_q, 16)

    def test_custom_out_dim(self) -> None:
        """Custom out_dim must be reflected in the output tensor."""
        out_dim = 48
        model = CrossAttention(
            query_dim=self.QUERY_DIM, kv_dim=self.KV_DIM, out_dim=out_dim
        )
        query = torch.randn(self.BATCH, 10, self.QUERY_DIM)
        key_value = torch.randn(self.BATCH, 20, self.KV_DIM)
        out = model(query, key_value)
        assert out.shape == (self.BATCH, 10, out_dim)


# ---------------------------------------------------------------------------
# MultiHeadAttention Tests
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    """Exhaustive tests for MultiHeadAttention."""

    BATCH = 2
    EMBED_DIM = 64
    NUM_HEADS = 8
    SEQ_LEN = 16

    def _model(self) -> MultiHeadAttention:
        return MultiHeadAttention(
            embed_dim=self.EMBED_DIM,
            num_heads=self.NUM_HEADS,
        )

    def _input(self) -> torch.Tensor:
        torch.manual_seed(10)
        return torch.randn(self.BATCH, self.SEQ_LEN, self.EMBED_DIM)

    # ---- Shape tests -------------------------------------------------------

    def test_output_shape(self) -> None:
        """Output shape must equal input shape (B, T, embed_dim)."""
        model = self._model()
        x = self._input()
        out = model(x)
        assert out.shape == (self.BATCH, self.SEQ_LEN, self.EMBED_DIM), (
            f"Expected {(self.BATCH, self.SEQ_LEN, self.EMBED_DIM)}, got {out.shape}"
        )

    # ---- Head-dimension correctness ----------------------------------------

    def test_head_dim_correctness(self) -> None:
        """head_dim must equal embed_dim // num_heads."""
        model = self._model()
        expected_head_dim = self.EMBED_DIM // self.NUM_HEADS
        assert model.head_dim == expected_head_dim, (
            f"Expected head_dim={expected_head_dim}, got {model.head_dim}"
        )

    def test_internal_reshape_no_error(self) -> None:
        """The split/merge head reshaping must execute without RuntimeError."""
        model = self._model()
        x = self._input()
        # Manually probe the private helpers
        projected = model.W_q(x)  # (B, T, embed_dim)
        split = model._split_heads(projected)  # (B, H, T, head_dim)
        assert split.shape == (
            self.BATCH, self.NUM_HEADS, self.SEQ_LEN, model.head_dim
        ), f"split shape mismatch: {split.shape}"

        merged = model._merge_heads(split)  # (B, T, embed_dim)
        assert merged.shape == (self.BATCH, self.SEQ_LEN, self.EMBED_DIM), (
            f"merge shape mismatch: {merged.shape}"
        )

    def test_invalid_num_heads_raises(self) -> None:
        """embed_dim not divisible by num_heads must raise ValueError."""
        with pytest.raises(ValueError, match="must be divisible by"):
            MultiHeadAttention(embed_dim=33, num_heads=8)

    # ---- Different configurations ------------------------------------------

    @pytest.mark.parametrize(
        ("embed_dim", "num_heads", "seq_len", "batch"),
        [
            (32, 4, 10, 1),
            (128, 8, 20, 4),
            (256, 16, 5, 2),
        ],
    )
    def test_various_configs(
        self, embed_dim: int, num_heads: int, seq_len: int, batch: int
    ) -> None:
        """Verify shape correctness across multiple (embed_dim, num_heads) configs."""
        model = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        x = torch.randn(batch, seq_len, embed_dim)
        out = model(x)
        assert out.shape == (batch, seq_len, embed_dim), (
            f"Config embed={embed_dim}, heads={num_heads}: "
            f"expected {(batch, seq_len, embed_dim)}, got {out.shape}"
        )

    # ---- Causal mask -------------------------------------------------------

    def test_causal_mask_accepted(self) -> None:
        """MultiHeadAttention must accept a 2-D causal mask without error."""
        model = self._model()
        x = self._input()
        T = self.SEQ_LEN
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        out = model(x, mask=mask)
        assert out.shape == (self.BATCH, self.SEQ_LEN, self.EMBED_DIM)


# ---------------------------------------------------------------------------
# SpatialAttention Tests
# ---------------------------------------------------------------------------

class TestSpatialAttention:
    """Exhaustive tests for SpatialAttention."""

    BATCH = 2
    CHANNELS = 16
    HEIGHT = 8
    WIDTH = 8

    def _model(self) -> SpatialAttention:
        return SpatialAttention(channels=self.CHANNELS)

    def _input(self) -> torch.Tensor:
        torch.manual_seed(20)
        return torch.randn(self.BATCH, self.CHANNELS, self.HEIGHT, self.WIDTH)

    # ---- Shape preservation ------------------------------------------------

    def test_output_shape_preserved(self) -> None:
        """Output must have exactly the same shape as the 4-D input."""
        model = self._model()
        x = self._input()
        out = model(x)
        assert out.shape == (self.BATCH, self.CHANNELS, self.HEIGHT, self.WIDTH), (
            f"Expected {(self.BATCH, self.CHANNELS, self.HEIGHT, self.WIDTH)}, "
            f"got {out.shape}"
        )

    def test_non_square_spatial_dims(self) -> None:
        """Spatial attention must handle non-square (H ≠ W) feature maps."""
        H, W = 6, 10
        model = SpatialAttention(channels=self.CHANNELS)
        x = torch.randn(self.BATCH, self.CHANNELS, H, W)
        out = model(x)
        assert out.shape == (self.BATCH, self.CHANNELS, H, W), (
            f"Expected {(self.BATCH, self.CHANNELS, H, W)}, got {out.shape}"
        )

    @pytest.mark.parametrize(
        ("batch", "channels", "h", "w"),
        [
            (1, 8, 4, 4),
            (4, 32, 16, 16),
            (2, 64, 7, 9),
        ],
    )
    def test_various_spatial_configs(
        self, batch: int, channels: int, h: int, w: int
    ) -> None:
        """Output spatial shape must survive any (B, C, H, W) configuration."""
        model = SpatialAttention(channels=channels)
        x = torch.randn(batch, channels, h, w)
        out = model(x)
        assert out.shape == (batch, channels, h, w), (
            f"Expected {(batch, channels, h, w)}, got {out.shape}"
        )

    # ---- Contiguity and dtype ----------------------------------------------

    def test_output_is_contiguous(self) -> None:
        """Output tensor must be contiguous after the reshape."""
        model = self._model()
        x = self._input()
        out = model(x)
        assert out.is_contiguous(), "Output tensor is not contiguous."

    def test_output_dtype_float32(self) -> None:
        """Output dtype must remain float32."""
        model = self._model()
        x = self._input()
        out = model(x)
        assert out.dtype == torch.float32

    # ---- Output differs from input (model actually transforms) -------------

    def test_output_differs_from_input(self) -> None:
        """The attention module must transform the input (not an identity fn)."""
        torch.manual_seed(30)
        model = self._model()
        model.eval()
        x = self._input()
        with torch.no_grad():
            out = model(x)
        assert not torch.allclose(out, x, atol=1e-6), (
            "SpatialAttention output is identical to input — "
            "the module may be a no-op."
        )
