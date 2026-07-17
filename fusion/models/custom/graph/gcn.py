"""
Graph Convolutional Network (GCN) encoder module.

Usage:
    from fusion.models.custom.graph import GCNEncoder
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv, global_mean_pool

    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False


class GCNEncoder(nn.Module):
    """Stacked GCN encoder with global mean pooling.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden features per layer.
        out_channels (int): Number of output features (graph embedding dim).
        num_layers (int): Total number of GCNConv layers (must be >= 2).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
    ) -> None:
        if not _TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for graph models. "
                "Install with: pip install torch-geometric"
            )
        if num_layers < 1:
            raise ValueError("`num_layers` must be >= 1.")

        super().__init__()

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()

        # Build layer list: [in → hidden, hidden → hidden, ..., hidden → out]
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(in_ch, out_ch))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node feature matrix of shape (N, in_channels).
            edge_index: Graph connectivity of shape (2, E).
            batch: Batch vector of shape (N,) assigning each node to a graph.
                   If None, all nodes are treated as a single graph.

        Returns:
            Graph-level embeddings of shape (B, out_channels).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.relu(x)

        # Global mean pooling → (B, out_channels)
        return global_mean_pool(x, batch)
