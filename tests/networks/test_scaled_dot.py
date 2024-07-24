import torch
from torch.nn.functional import scaled_dot_product_attention

from continuiti.networks import ScaledDotProductAttention


class TestScaledDotProductAttention:
    batch_size = 3
    query_size = 5
    key_val_size = 7
    hidden_dim = 11

    def test_forward_correct(self):
        query = torch.rand(self.batch_size, self.query_size, self.hidden_dim)
        key = torch.rand(self.batch_size, self.key_val_size, self.hidden_dim)
        value = torch.rand(self.batch_size, self.key_val_size, self.hidden_dim)

        attn = ScaledDotProductAttention()

        out = attn(query, key, value)
        gt_out = scaled_dot_product_attention(query, key, value)

        assert torch.allclose(out, gt_out)

    def test_masked_correct(self):
        query = torch.rand(self.batch_size, self.query_size, self.hidden_dim)
        key = torch.rand(self.batch_size, self.key_val_size, self.hidden_dim)
        value = torch.rand(self.batch_size, self.key_val_size, self.hidden_dim)
        mask = torch.rand(self.batch_size, self.key_val_size) >= 0.2

        attn = ScaledDotProductAttention()

        out = attn(query, key, value, mask)

        gt_mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
        out_gt = scaled_dot_product_attention(query, key, value, gt_mask)

        assert torch.allclose(out, out_gt)
