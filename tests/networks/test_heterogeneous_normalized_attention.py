import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from continuiti.networks import HeterogeneousNormalizedAttention


@pytest.fixture(scope="module")
def random_query_key_value_pair():
    batch_size = 3
    query_size = 5
    key_val_size = 7
    hidden_dim = 11

    query = torch.rand(batch_size, query_size, hidden_dim)
    key = torch.rand(batch_size, key_val_size, hidden_dim)
    value = torch.rand(batch_size, key_val_size, hidden_dim)

    return query, key, value


class TestHeterogeneousNormalized:
    def test_can_initialize(self):
        _ = HeterogeneousNormalizedAttention()
        assert True

    def test_shape_correct(self, random_query_key_value_pair):
        query, key, value = random_query_key_value_pair

        attn = HeterogeneousNormalizedAttention()

        out = attn(query, key, value)
        gt_out = scaled_dot_product_attention(query, key, value)

        assert out.shape == gt_out.shape

    def test_gradient_flow(self, random_query_key_value_pair):
        query, key, value = random_query_key_value_pair
        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True

        attn = HeterogeneousNormalizedAttention()

        out = attn(query, key, value)

        out.sum().backward()

        assert query.grad is not None, "Gradients not flowing to query"
        assert key.grad is not None, "Gradients not flowing to key"
        assert value.grad is not None, "Gradients not flowing to value"

    def test_zero_input(self, random_query_key_value_pair):
        query, key, value = random_query_key_value_pair
        attn = HeterogeneousNormalizedAttention()
        out = attn(query, key, torch.zeros(value.shape))
        assert torch.allclose(torch.zeros(out.shape), out)

    def test_mask_forward(self, random_query_key_value_pair):
        query, key, value = random_query_key_value_pair
        attn = HeterogeneousNormalizedAttention()

        # masks in the operator setting should be always block tensors with the upper left block of the last two
        # dimensions being True. The dimensions of the True block corresponds to the numbers of sensors and evaluations.
        mask = []
        mask = torch.rand(query.size(0), key.size(1)) >= 0.2

        out = attn(query, key, value, mask)

        assert isinstance(out, torch.Tensor)

    def test_mask_correct(self, random_query_key_value_pair):
        query, key, value = random_query_key_value_pair
        attn = HeterogeneousNormalizedAttention()

        out_gt = attn(query, key, value)

        key_rand = torch.rand(key.shape)
        key_masked = torch.cat([key, key_rand], dim=1)

        value_rand = torch.rand(value.shape)
        value_masked = torch.cat([value, value_rand], dim=1)

        true_mask = torch.ones(value.size(0), value.size(1), dtype=torch.bool)
        attn_mask = torch.cat([true_mask, ~true_mask], dim=1)

        out_masked = attn(query, key_masked, value_masked, attn_mask)

        assert torch.allclose(out_gt, out_masked)
