# pytest tests/ -x -v

import torch
import pytest
from unittest.mock import patch

try:
    import torch_geometric
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from fusion.models.custom.graph.gcn import GCNEncoder
from fusion.models.custom.graph.gat import GATEncoder
from fusion.models.custom.graph.graph_sage import GraphSAGEEncoder
from fusion.models.custom.graph.gin import GINEncoder


def test_missing_pyg_error(monkeypatch):
    """
    Ensure that an ImportError is correctly thrown during instantiation
    if torch_geometric is missing (simulated via monkeypatching).
    """
    import fusion.models.custom.graph.gcn as gcn_mod
    import fusion.models.custom.graph.gat as gat_mod
    import fusion.models.custom.graph.graph_sage as sage_mod
    import fusion.models.custom.graph.gin as gin_mod
    
    # Temporarily set HAS_PYG to False in GNN modules
    monkeypatch.setattr(gcn_mod, "HAS_PYG", False)
    monkeypatch.setattr(gat_mod, "HAS_PYG", False)
    monkeypatch.setattr(sage_mod, "HAS_PYG", False)
    monkeypatch.setattr(gin_mod, "HAS_PYG", False)
    
    with pytest.raises(ImportError, match="torch_geometric is required"):
        gcn_mod.GCNEncoder(in_channels=16, hidden_channels=8, out_channels=4, num_layers=2)
        
    with pytest.raises(ImportError, match="torch_geometric is required"):
        gat_mod.GATEncoder(in_channels=16, hidden_channels=8, out_channels=4, num_layers=2)
        
    with pytest.raises(ImportError, match="torch_geometric is required"):
        sage_mod.GraphSAGEEncoder(in_channels=16, hidden_channels=8, out_channels=4, num_layers=2)
        
    with pytest.raises(ImportError, match="torch_geometric is required"):
        gin_mod.GINEncoder(in_channels=16, hidden_channels=8, out_channels=4, num_layers=2)


@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric is not installed in the environment")
def test_gnn_basic_forward_pass():
    """
    Test GCN, GAT, and GIN encoders forward pass and assert exact output shapes.
    """
    # 10 nodes, 16 features each
    x = torch.randn(10, 16)
    # Directed edge list (10 edges)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    ], dtype=torch.long)
    # Batch vector indicating node to graph mapping (2 graphs, batch_size=2)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    
    # 1. Test GCNEncoder
    gcn = GCNEncoder(in_channels=16, hidden_channels=8, out_channels=4, num_layers=3)
    out_gcn = gcn(x, edge_index, batch)
    assert out_gcn.shape == (2, 4)  # (batch_size, out_channels)
    
    # 2. Test GATEncoder
    gat = GATEncoder(in_channels=16, hidden_channels=8, out_channels=4, num_layers=2, heads=2)
    out_gat = gat(x, edge_index, batch)
    assert out_gat.shape == (2, 4)  # (batch_size, out_channels)
    
    # 3. Test GINEncoder
    gin = GINEncoder(in_channels=16, hidden_channels=8, out_channels=4, num_layers=2, eps=0.1, train_eps=True)
    out_gin = gin(x, edge_index, batch)
    assert out_gin.shape == (2, 4)  # (batch_size, out_channels)


@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric is not installed in the environment")
def test_graph_sage_configurations():
    """
    Specifically test GraphSAGEEncoder by iterating through all three aggregation modes
    ("mean", "max", "lstm") to ensure they all initialize and pass properly.
    """
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    ], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    
    for aggr in ["mean", "max", "lstm"]:
        # Test SAGE with batchnorm
        sage = GraphSAGEEncoder(
            in_channels=16, 
            hidden_channels=8, 
            out_channels=4, 
            num_layers=2, 
            aggr=aggr, 
            use_batchnorm=True
        )
        out = sage(x, edge_index, batch)
        assert out.shape == (2, 4)
        
        # Test SAGE without batchnorm
        sage_no_bn = GraphSAGEEncoder(
            in_channels=16, 
            hidden_channels=8, 
            out_channels=4, 
            num_layers=2, 
            aggr=aggr, 
            use_batchnorm=False
        )
        out_no_bn = sage_no_bn(x, edge_index, batch)
        assert out_no_bn.shape == (2, 4)
