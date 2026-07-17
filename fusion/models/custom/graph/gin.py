"""
Graph Isomorphism Network (GIN) encoder module.

Usage:
    from fusion.models.custom.graph import GINEncoder
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GINConv, global_mean_pool

    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False


def _build_mlp(in_channels: int, out_channels: int) -> nn.Sequential:
    """Construct the two-layer MLP used inside GINConv.

    Architecture: Linear → BatchNorm → ReLU → Linear → ReLU

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.

    Returns:
        A ``nn.Sequential`` MLP module.
    """
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Linear(out_channels, out_channels),
        nn.ReLU(),
    )


class GINEncoder(nn.Module):
    """Stacked GIN encoder with MLP aggregation and global mean pooling.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden features per layer.
        out_channels (int): Number of output graph embedding features.
        num_layers (int): Total number of GINConv layers (must be >= 1).
        eps (float): Initial value for the ``ε`` parameter.
        train_eps (bool): If ``True``, ``ε`` is a learnable parameter;
            otherwise it is a fixed constant.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        eps: float = 0.0,
        train_eps: bool = False,
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

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            mlp = _build_mlp(in_ch, out_ch)
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))

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

        for conv in self.convs:
            x = conv(x, edge_index)

        return global_mean_pool(x, batch)
