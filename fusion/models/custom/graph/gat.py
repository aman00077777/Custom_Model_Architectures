import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    import torch_geometric
    from torch_geometric.nn import GATConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GATEncoder(nn.Module):
    """
    GATEncoder model that stacks GATConv layers with multi-head attention.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, heads: int = 1):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for graph models. Install with: pip install torch-geometric")
        
        if num_layers <= 0:
            raise ValueError("num_layers must be at least 1")
            
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=heads, concat=False))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False))
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
        x = global_mean_pool(x, batch)
        return x
