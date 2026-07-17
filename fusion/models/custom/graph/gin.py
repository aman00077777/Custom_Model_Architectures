import torch
import torch.nn as nn
from typing import Optional

try:
    import torch_geometric
    from torch_geometric.nn import GINConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GINEncoder(nn.Module):
    """
    GINEncoder model that stacks GINConv layers with MLPs.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int, eps: float = 0.0, train_eps: bool = False):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for graph models. Install with: pip install torch-geometric")
        
        if num_layers <= 0:
            raise ValueError("num_layers must be at least 1")
            
        self.convs = nn.ModuleList()
        
        if num_layers == 1:
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))
        else:
            # Layer 1
            mlp1 = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp1, eps=eps, train_eps=train_eps))
            # Intermediate layers
            for _ in range(num_layers - 2):
                mlp_mid = nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GINConv(mlp_mid, eps=eps, train_eps=train_eps))
            # Final layer
            mlp_fin = nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            )
            self.convs.append(GINConv(mlp_fin, eps=eps, train_eps=train_eps))
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        x = global_mean_pool(x, batch)
        return x
