import math
import torch
import torch.nn as nn
from typing import Optional


class CrossAttention(nn.Module):
    """
    CrossAttention module that computes attention between query and key_value tensors.
    """
    def __init__(self, query_dim: int, kv_dim: int, embed_dim: Optional[int] = None):
        super().__init__()
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.embed_dim = embed_dim if embed_dim is not None else query_dim
        
        self.q_proj = nn.Linear(query_dim, self.embed_dim)
        self.k_proj = nn.Linear(kv_dim, self.embed_dim)
        self.v_proj = nn.Linear(kv_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, query_dim)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query shape: (B, Sq, Cq)
        # key_value shape: (B, Skv, Ckv)
        B, Sq, Cq = query.shape
        _, Skv, Ckv = key_value.shape
        
        q = self.q_proj(query)      # (B, Sq, embed_dim)
        k = self.k_proj(key_value)  # (B, Skv, embed_dim)
        v = self.v_proj(key_value)  # (B, Skv, embed_dim)
        
        scale = 1.0 / math.sqrt(self.embed_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, Sq, Skv)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                scores = scores + mask
                
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)  # (B, Sq, embed_dim)
        out = self.out_proj(out)             # (B, Sq, query_dim)
        return out
