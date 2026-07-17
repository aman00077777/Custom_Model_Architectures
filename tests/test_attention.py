# pytest tests/ -x -v

import torch
import pytest
from fusion.models.custom.attention.self_attention import SelfAttention
from fusion.models.custom.attention.cross_attention import CrossAttention
from fusion.models.custom.attention.multi_head_attention import MultiHeadAttention
from fusion.models.custom.attention.spatial_attention import SpatialAttention


def test_self_attention():
    B, S, E = 2, 5, 8
    model = SelfAttention(embed_dim=E, dropout=0.0)
    x = torch.randn(B, S, E)
    
    # Test without mask
    out_no_mask = model(x)
    assert out_no_mask.shape == (B, S, E)
    
    # Test with causal mask
    out_causal = model(x, causal=True)
    assert out_causal.shape == (B, S, E)
    
    # Verify causal masking affects values (first token output should be independent of subsequent tokens)
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] = x_perturbed[:, -1, :] + 10.0
    
    out_orig_causal = model(x, causal=True)
    out_pert_causal = model(x_perturbed, causal=True)
    
    # The output at the first sequence position should be exactly the same, since it cannot attend to the last token
    assert torch.allclose(out_orig_causal[:, 0, :], out_pert_causal[:, 0, :], atol=1e-5)
    
    # However, without causal masking, the output at the first position will change
    out_orig_no_mask = model(x)
    out_pert_no_no_mask = model(x_perturbed)
    assert not torch.allclose(out_orig_no_mask[:, 0, :], out_pert_no_no_mask[:, 0, :], atol=1e-5)


def test_cross_attention():
    B = 2
    Sq, Skv = 10, 20
    query_dim, kv_dim = 16, 32
    embed_dim = 24
    
    model = CrossAttention(query_dim=query_dim, kv_dim=kv_dim, embed_dim=embed_dim)
    
    query = torch.randn(B, Sq, query_dim)
    key_value = torch.randn(B, Skv, kv_dim)
    
    # Test forward pass with sequence length mismatch
    out = model(query, key_value)
    assert out.shape == (B, Sq, query_dim)
    
    # Test with boolean mask of shape (B, Skv)
    mask = torch.ones(B, Skv, dtype=torch.bool)
    mask[:, -5:] = False  # Mask the last 5 positions
    
    out_masked = model(query, key_value, mask=mask)
    assert out_masked.shape == (B, Sq, query_dim)


def test_multi_head_attention():
    B, S, E = 4, 12, 32
    num_heads = 4
    model = MultiHeadAttention(embed_dim=E, num_heads=num_heads, dropout=0.1)
    
    x = torch.randn(B, S, E)
    out = model(x)
    assert out.shape == (B, S, E)
    
    # Test with causal mask
    out_causal = model(x, causal=True)
    assert out_causal.shape == (B, S, E)
    
    # Check that error is raised for invalid embed_dim / num_heads combination
    with pytest.raises(ValueError):
        MultiHeadAttention(embed_dim=15, num_heads=2)


def test_spatial_attention():
    B, C, H, W = 2, 16, 8, 8
    model = SpatialAttention(channels=C, dropout=0.0)
    
    x = torch.randn(B, C, H, W)
    out = model(x)
    
    # Assert output shape matches input shape exactly
    assert out.shape == (B, C, H, W)
