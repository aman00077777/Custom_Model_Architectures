# Run with: pytest tests/ -x -v
"""
Strict TDD test suite for Graph Neural Network modules.

Execute:
    pytest tests/ -x -v

The -x flag halts immediately on the first failure.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Availability probe — used for skipif markers
# ---------------------------------------------------------------------------
try:
    import torch_geometric  # noqa: F401

    _TG_AVAILABLE = True
except ImportError:
    _TG_AVAILABLE = False

_SKIP_IF_NO_TG = pytest.mark.skipif(
    not _TG_AVAILABLE,
    reason="torch_geometric is not installed — skipping forward-pass tests.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(
    num_nodes: int = 20,
    in_channels: int = 16,
    num_graphs: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a dummy graph batch.

    Returns:
        x: Node features (num_nodes, in_channels).
        edge_index: Random edges (2, E).
        batch: Batch assignment vector (num_nodes,).
    """
    torch.manual_seed(42)
    x = torch.randn(num_nodes, in_channels)

    # Random sparse edge_index
    src = torch.randint(0, num_nodes, (num_nodes * 2,))
    dst = torch.randint(0, num_nodes, (num_nodes * 2,))
    edge_index = torch.stack([src, dst], dim=0)

    # Assign nodes to graphs round-robin
    batch = torch.arange(num_nodes) % num_graphs

    return x, edge_index, batch


# ---------------------------------------------------------------------------
# Forward-pass tests (skipped without torch_geometric)
# ---------------------------------------------------------------------------

@_SKIP_IF_NO_TG
class TestGCNEncoderForward:
    """GCNEncoder basic forward-pass tests."""

    IN_CH = 16
    HIDDEN_CH = 32
    OUT_CH = 8
    NUM_GRAPHS = 4

    def test_output_shape_default_layers(self) -> None:
        """Output must be (batch_size, out_channels) with num_layers=3."""
        from fusion.models.custom.graph import GCNEncoder

        model = GCNEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            num_layers=3,
        )
        model.eval()
        x, edge_index, batch = _make_batch(
            num_nodes=20, in_channels=self.IN_CH, num_graphs=self.NUM_GRAPHS
        )
        with torch.no_grad():
            out = model(x, edge_index, batch)

        assert out.shape == (self.NUM_GRAPHS, self.OUT_CH), (
            f"Expected ({self.NUM_GRAPHS}, {self.OUT_CH}), got {out.shape}"
        )

    def test_output_shape_single_layer(self) -> None:
        """GCNEncoder with num_layers=1 must still produce valid output."""
        from fusion.models.custom.graph import GCNEncoder

        model = GCNEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            num_layers=1,
        )
        model.eval()
        x, edge_index, batch = _make_batch(
            num_nodes=20, in_channels=self.IN_CH, num_graphs=self.NUM_GRAPHS
        )
        with torch.no_grad():
            out = model(x, edge_index, batch)

        assert out.shape == (self.NUM_GRAPHS, self.OUT_CH)

    def test_output_is_float_tensor(self) -> None:
        """Output dtype must be float32."""
        from fusion.models.custom.graph import GCNEncoder

        model = GCNEncoder(self.IN_CH, self.HIDDEN_CH, self.OUT_CH, num_layers=2)
        model.eval()
        x, edge_index, batch = _make_batch(
            num_nodes=20, in_channels=self.IN_CH, num_graphs=self.NUM_GRAPHS
        )
        with torch.no_grad():
            out = model(x, edge_index, batch)

        assert out.dtype == torch.float32


@_SKIP_IF_NO_TG
class TestGATEncoderForward:
    """GATEncoder basic forward-pass tests."""

    IN_CH = 16
    HIDDEN_CH = 8
    OUT_CH = 4
    HEADS = 4
    NUM_GRAPHS = 3

    def test_output_shape(self) -> None:
        """Output must be (batch_size, out_channels)."""
        from fusion.models.custom.graph import GATEncoder

        model = GATEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            num_layers=3,
            heads=self.HEADS,
        )
        model.eval()
        x, edge_index, batch = _make_batch(
            num_nodes=15, in_channels=self.IN_CH, num_graphs=self.NUM_GRAPHS
        )
        with torch.no_grad():
            out = model(x, edge_index, batch)

        assert out.shape == (self.NUM_GRAPHS, self.OUT_CH), (
            f"Expected ({self.NUM_GRAPHS}, {self.OUT_CH}), got {out.shape}"
        )

    def test_single_layer_output_shape(self) -> None:
        """Single-layer GAT should skip concat logic and go straight to out."""
        from fusion.models.custom.graph import GATEncoder

        model = GATEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            num_layers=1,
            heads=self.HEADS,
        )
        model.eval()
        x, edge_index, batch = _make_batch(
            num_nodes=15, in_channels=self.IN_CH, num_graphs=self.NUM_GRAPHS
        )
        with torch.no_grad():
            out = model(x, edge_index, batch)

        assert out.shape == (self.NUM_GRAPHS, self.OUT_CH)


@_SKIP_IF_NO_TG
class TestGraphSAGEEncoderForward:
    """GraphSAGEEncoder basic forward-pass tests."""

    IN_CH = 16
    HIDDEN_CH = 32
    OUT_CH = 8
    NUM_GRAPHS = 4

    def test_output_shape_mean_aggr(self) -> None:
        from fusion.models.custom.graph import GraphSAGEEncoder

        model = GraphSAGEEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            aggr="mean",
        )
        model.eval()
        x, edge_index, batch = _make_batch(
            num_nodes=20, in_channels=self.IN_CH, num_graphs=self.NUM_GRAPHS
        )
        with torch.no_grad():
            out = model(x, edge_index, batch)

        assert out.shape == (self.NUM_GRAPHS, self.OUT_CH)

    def test_batchnorm_enabled(self) -> None:
        """use_batchnorm=True must not alter the output shape."""
        from fusion.models.custom.graph import GraphSAGEEncoder

        model = GraphSAGEEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            aggr="mean",
            use_batchnorm=True,
        )
        # train mode required for BatchNorm with batch > 1
        x, edge_index, batch = _make_batch(
            num_nodes=20, in_channels=self.IN_CH, num_graphs=self.NUM_GRAPHS
        )
        out = model(x, edge_index, batch)

        assert out.shape == (self.NUM_GRAPHS, self.OUT_CH)


@_SKIP_IF_NO_TG
class TestGraphSAGEAggregationModes:
    """Verify all three aggregation modes initialize and forward correctly."""

    IN_CH = 16
    HIDDEN_CH = 32
    OUT_CH = 8
    NUM_GRAPHS = 4
    NUM_NODES = 20

    @pytest.mark.parametrize("aggr", ["mean", "max", "lstm"])
    def test_aggr_mode(self, aggr: str) -> None:
        """Each aggregation mode must produce (batch_size, out_channels).

        Note: PyG's LSTM aggregator requires edge_index sorted by destination
        node (row=1). We sort by dst before forwarding.
        """
        from fusion.models.custom.graph import GraphSAGEEncoder

        model = GraphSAGEEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            num_layers=2,
            aggr=aggr,  # type: ignore[arg-type]
        )
        model.eval()
        x, edge_index, batch = _make_batch(
            num_nodes=self.NUM_NODES,
            in_channels=self.IN_CH,
            num_graphs=self.NUM_GRAPHS,
        )

        # LSTM aggregation requires edge_index sorted by destination node
        if aggr == "lstm":
            sort_idx = edge_index[1].argsort()
            edge_index = edge_index[:, sort_idx]

        with torch.no_grad():
            out = model(x, edge_index, batch)

        assert out.shape == (self.NUM_GRAPHS, self.OUT_CH), (
            f"aggr='{aggr}': expected ({self.NUM_GRAPHS}, {self.OUT_CH}), "
            f"got {out.shape}"
        )


@_SKIP_IF_NO_TG
class TestGINEncoderForward:
    """GINEncoder basic forward-pass tests."""

    IN_CH = 16
    HIDDEN_CH = 32
    OUT_CH = 8
    NUM_GRAPHS = 4

    def test_output_shape_fixed_eps(self) -> None:
        """Fixed eps should produce correct output shape."""
        from fusion.models.custom.graph import GINEncoder

        model = GINEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            num_layers=3,
            train_eps=False,
        )
        model.eval()
        x, edge_index, batch = _make_batch(
            num_nodes=20, in_channels=self.IN_CH, num_graphs=self.NUM_GRAPHS
        )
        with torch.no_grad():
            out = model(x, edge_index, batch)

        assert out.shape == (self.NUM_GRAPHS, self.OUT_CH)

    def test_output_shape_learnable_eps(self) -> None:
        """Learnable eps should produce correct output shape."""
        from fusion.models.custom.graph import GINEncoder

        model = GINEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            num_layers=3,
            train_eps=True,
        )
        x, edge_index, batch = _make_batch(
            num_nodes=20, in_channels=self.IN_CH, num_graphs=self.NUM_GRAPHS
        )
        out = model(x, edge_index, batch)

        assert out.shape == (self.NUM_GRAPHS, self.OUT_CH)

    def test_learnable_eps_is_parameter(self) -> None:
        """When train_eps=True, eps must appear in model.parameters()."""
        from fusion.models.custom.graph import GINEncoder

        model = GINEncoder(
            in_channels=self.IN_CH,
            hidden_channels=self.HIDDEN_CH,
            out_channels=self.OUT_CH,
            num_layers=1,
            train_eps=True,
        )
        param_names = [name for name, _ in model.named_parameters()]
        eps_params = [n for n in param_names if "eps" in n]
        assert len(eps_params) > 0, (
            "Expected at least one learnable eps parameter, found none."
        )


# ---------------------------------------------------------------------------
# Error-handling tests — must run even without torch_geometric
# ---------------------------------------------------------------------------

class TestImportErrorHandling:
    """Ensure ImportError is raised when torch_geometric is missing."""

    def _patch_tg_absent(self) -> dict[str, None]:
        """Return a sys.modules patch that removes torch_geometric."""
        return {
            "torch_geometric": None,
            "torch_geometric.nn": None,
        }

    def _reimport_module(self, module_path: str) -> types.ModuleType:
        """Force-reimport a module after clearing it from sys.modules."""
        # Remove the target and any sub-modules from the cache
        keys_to_remove = [
            k for k in list(sys.modules.keys())
            if k == module_path or k.startswith(module_path + ".")
        ]
        for k in keys_to_remove:
            del sys.modules[k]
        return importlib.import_module(module_path)

    def test_gcn_raises_import_error(self) -> None:
        """GCNEncoder.__init__ must raise ImportError when tg is absent."""
        with patch.dict(sys.modules, {"torch_geometric": None, "torch_geometric.nn": None}):
            mod = self._reimport_module("fusion.models.custom.graph.gcn")
            with pytest.raises(ImportError, match="torch_geometric is required"):
                mod.GCNEncoder(
                    in_channels=8, hidden_channels=16, out_channels=4
                )

    def test_gat_raises_import_error(self) -> None:
        """GATEncoder.__init__ must raise ImportError when tg is absent."""
        with patch.dict(sys.modules, {"torch_geometric": None, "torch_geometric.nn": None}):
            mod = self._reimport_module("fusion.models.custom.graph.gat")
            with pytest.raises(ImportError, match="torch_geometric is required"):
                mod.GATEncoder(
                    in_channels=8, hidden_channels=4, out_channels=4
                )

    def test_graph_sage_raises_import_error(self) -> None:
        """GraphSAGEEncoder.__init__ must raise ImportError when tg is absent."""
        with patch.dict(sys.modules, {"torch_geometric": None, "torch_geometric.nn": None}):
            mod = self._reimport_module("fusion.models.custom.graph.graph_sage")
            with pytest.raises(ImportError, match="torch_geometric is required"):
                mod.GraphSAGEEncoder(
                    in_channels=8, hidden_channels=16, out_channels=4
                )

    def test_gin_raises_import_error(self) -> None:
        """GINEncoder.__init__ must raise ImportError when tg is absent."""
        with patch.dict(sys.modules, {"torch_geometric": None, "torch_geometric.nn": None}):
            mod = self._reimport_module("fusion.models.custom.graph.gin")
            with pytest.raises(ImportError, match="torch_geometric is required"):
                mod.GINEncoder(
                    in_channels=8, hidden_channels=16, out_channels=4
                )

    def test_error_message_contains_install_hint(self) -> None:
        """The ImportError message must include the pip install hint."""
        with patch.dict(sys.modules, {"torch_geometric": None, "torch_geometric.nn": None}):
            mod = self._reimport_module("fusion.models.custom.graph.gcn")
            with pytest.raises(ImportError, match="pip install torch-geometric"):
                mod.GCNEncoder(
                    in_channels=8, hidden_channels=16, out_channels=4
                )


# ---------------------------------------------------------------------------
# Configuration validation tests
# ---------------------------------------------------------------------------

class TestGraphSAGEConfigValidation:
    """Test invalid config values raise expected exceptions."""

    @pytest.mark.skipif(not _TG_AVAILABLE, reason="torch_geometric not installed")
    def test_invalid_aggr_raises_value_error(self) -> None:
        from fusion.models.custom.graph import GraphSAGEEncoder

        with pytest.raises(ValueError, match="`aggr` must be one of"):
            GraphSAGEEncoder(
                in_channels=8, hidden_channels=16, out_channels=4, aggr="sum"  # type: ignore[arg-type]
            )

    @pytest.mark.skipif(not _TG_AVAILABLE, reason="torch_geometric not installed")
    def test_zero_layers_raises_value_error(self) -> None:
        from fusion.models.custom.graph import GCNEncoder

        with pytest.raises(ValueError, match="`num_layers` must be >= 1"):
            GCNEncoder(
                in_channels=8, hidden_channels=16, out_channels=4, num_layers=0
            )
