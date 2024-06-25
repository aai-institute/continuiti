import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention
from random import randint

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
        for batch in range(query.size(0)):
            n_q = randint(1, query.size(1) - 1)
            n_kv = randint(1, key.size(1) - 1)

            ul = torch.ones(n_q, n_kv, dtype=torch.bool)
            ur = torch.zeros(n_q, key.size(1) - n_kv, dtype=torch.bool)
            u = torch.cat([ul, ur], dim=1)

            bl = torch.zeros(query.size(1) - n_q, n_kv, dtype=torch.bool)
            br = torch.zeros(query.size(1) - n_q, key.size(1) - n_kv, dtype=torch.bool)
            b = torch.cat([bl, br], dim=1)

            mask.append(torch.cat([u, b], dim=0))

        mask = torch.stack(mask, dim=0)

        out = attn(query, key, value, mask)

        assert isinstance(out, torch.Tensor)
