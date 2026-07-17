"""
GraphSAGE encoder module.

Usage:
    from fusion.models.custom.graph import GraphSAGEEncoder
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import SAGEConv, global_mean_pool

    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False

AggregationType = Literal["mean", "max", "lstm"]


class GraphSAGEEncoder(nn.Module):
    """Stacked GraphSAGE encoder with configurable aggregation and BatchNorm.

    Args:
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden features per layer.
        out_channels (int): Number of output graph embedding features.
        num_layers (int): Total number of SAGEConv layers (must be >= 1).
        aggr (str): Neighbourhood aggregation method.
            One of ``"mean"``, ``"max"``, or ``"lstm"``.
        use_batchnorm (bool): If ``True``, apply BatchNorm1d after each layer.
        dropout (float): Dropout rate applied after activation in each layer.
    """

    _VALID_AGGR: tuple[str, ...] = ("mean", "max", "lstm")

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        aggr: AggregationType = "mean",
        use_batchnorm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        if not _TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for graph models. "
                "Install with: pip install torch-geometric"
            )
        if aggr not in self._VALID_AGGR:
            raise ValueError(
                f"`aggr` must be one of {self._VALID_AGGR}, got '{aggr}'."
            )
        if num_layers < 1:
            raise ValueError("`num_layers` must be >= 1.")

        super().__init__()

        self.convs = nn.ModuleList()
        self.bns: Optional[nn.ModuleList] = (
            nn.ModuleList() if use_batchnorm else None
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.use_batchnorm = use_batchnorm

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(in_ch, out_ch, aggr=aggr))
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(out_ch))  # type: ignore[union-attr]

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
            if self.use_batchnorm and self.bns is not None:
                x = self.bns[i](x)
            if i < len(self.convs) - 1:
                x = self.relu(x)
                x = self.dropout(x)

        return global_mean_pool(x, batch)
