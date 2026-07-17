import math
import torch
import torch.nn as nn
from typing import Optional


class SelfAttention(nn.Module):
    """
    SelfAttention module that computes scaled dot-product attention on input tensors.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, causal: bool = False) -> torch.Tensor:
        # x shape: (B, S, E)
        B, S, E = x.shape
        
        q = self.q_proj(x)  # (B, S, E)
        k = self.k_proj(x)  # (B, S, E)
        v = self.v_proj(x)  # (B, S, E)
        
        scale = 1.0 / math.sqrt(E)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, S, S)
        
        if causal:
            causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
            
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                scores = scores + mask
                
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)  # (B, S, E)
        out = self.out_proj(out)
        return out
