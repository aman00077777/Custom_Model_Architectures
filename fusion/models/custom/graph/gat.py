"""
Graph Attention Network (GAT) encoder module.

Usage:
    from fusion.models.custom.graph import GATEncoder
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv, global_mean_pool

    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False


class GATEncoder(nn.Module):
    """Stacked GAT encoder with multi-head attention and global mean pooling.

    Intermediate layers use ``concat=True`` (output dim = hidden * heads).
    The final layer uses ``concat=False`` (output dim = out_channels).

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden features per attention head.
        out_channels (int): Number of output graph embedding features.
        num_layers (int): Total number of GATConv layers (must be >= 1).
        heads (int): Number of attention heads for intermediate layers.
        dropout (float): Dropout probability inside GATConv.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
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
        self.elu = nn.ELU()

        for i in range(num_layers):
            is_final = i == num_layers - 1

            # Input channels depend on the previous layer's output
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = hidden_channels * heads  # previous concat output

            if is_final:
                self.convs.append(
                    GATConv(
                        in_ch,
                        out_channels,
                        heads=1,
                        concat=False,
                        dropout=dropout,
                    )
                )
            else:
                self.convs.append(
                    GATConv(
                        in_ch,
                        hidden_channels,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                    )
                )

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
            batch: Batch vector of shape (N,). Defaults to single-graph.

        Returns:
            Graph-level embeddings of shape (B, out_channels).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.elu(x)

        return global_mean_pool(x, batch)
