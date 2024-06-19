import torch
from torch.nn.functional import scaled_dot_product_attention

from continuiti.networks import ScaledDotProductAttention


def test_forward_correct():
    batch_size = 3
    query_size = 5
    key_val_size = 7
    hidden_dim = 11

    query = torch.rand(batch_size, query_size, hidden_dim)
    key = torch.rand(batch_size, key_val_size, hidden_dim)
    value = torch.rand(batch_size, key_val_size, hidden_dim)

    attn = ScaledDotProductAttention()

    out = attn(query, key, value)
    gt_out = scaled_dot_product_attention(query, key, value)

    assert torch.allclose(out, gt_out)
