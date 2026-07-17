import torch
import torch.nn as nn
from typing import Optional

try:
    import torch_geometric
    from torch_geometric.nn import SAGEConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGEEncoder model that stacks SAGEConv layers.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, aggr: str = "mean", use_batchnorm: bool = False):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for graph models. Install with: pip install torch-geometric")
        
        if num_layers <= 0:
            raise ValueError("num_layers must be at least 1")
        if aggr not in ["mean", "max", "lstm"]:
            raise ValueError(f"aggr must be 'mean', 'max', or 'lstm', got {aggr}")
            
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None
        
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
                if use_batchnorm:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(out_channels))
                
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            if self.bns is not None:
                x = self.bns[i](x)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return x
