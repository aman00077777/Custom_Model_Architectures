"""
Graph module public API.

Exported classes:
    - GCNEncoder
    - GATEncoder
    - GraphSAGEEncoder
    - GINEncoder
"""

from fusion.models.custom.graph.gcn import GCNEncoder
from fusion.models.custom.graph.gat import GATEncoder
from fusion.models.custom.graph.graph_sage import GraphSAGEEncoder
from fusion.models.custom.graph.gin import GINEncoder

__all__ = [
    "GCNEncoder",
    "GATEncoder",
    "GraphSAGEEncoder",
    "GINEncoder",
]
