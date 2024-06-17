import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention
from continuiti.networks.attention import HeterogeneousNormalized


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
    def test_shape_correct(self, random_query_key_value_pair):
        query, key, value = random_query_key_value_pair

        attn = HeterogeneousNormalized()

        out = attn(query, key, value)
        gt_out = scaled_dot_product_attention(query, key, value)

        assert out.shape == gt_out.shape
