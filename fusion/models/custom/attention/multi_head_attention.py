import math
import torch
import torch.nn as nn
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention module that reshapes Q, K, V into multiple heads
    and performs scaled dot-product attention per head.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
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
        
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, num_heads, S, S)
        
        if causal:
            causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
            
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)               # (B, 1, S, S)
                
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                scores = scores + mask
                
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)  # (B, num_heads, S, head_dim)
        
        out = out.transpose(1, 2).contiguous().view(B, S, E)
        out = self.out_proj(out)
        return out
