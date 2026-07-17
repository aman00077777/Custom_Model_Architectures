import torch
import torch.nn as nn
from typing import Optional

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GCNEncoder(nn.Module):
    """
    GCNEncoder model that stacks GCNConv layers.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for graph models. Install with: pip install torch-geometric")
        
        if num_layers <= 0:
            raise ValueError("num_layers must be at least 1")
            
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return x
