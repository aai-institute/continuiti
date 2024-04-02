import pytest
import torch
import torch.nn as nn

from continuiti.networks import MultiHeadAttention


@pytest.fixture(scope="session")
def some_multi_head_attn():
    return MultiHeadAttention(
        hidden_dim=32,
        n_heads=4,
        attention=nn.functional.scaled_dot_product_attention,
        dropout_p=0.25,
        bias=True,
    )


class TestMultiHeadAttention:
    def test_can_initialize(self, some_multi_head_attn):
        assert isinstance(some_multi_head_attn, MultiHeadAttention)

    def test_output_shape(self, some_multi_head_attn):
        batch_size = 3
        hidden_dim = 32
        query_size = 5
        key_val_size = 7

        query = torch.rand(batch_size, query_size, hidden_dim)
        key = torch.rand(batch_size, key_val_size, hidden_dim)
        val = torch.rand(batch_size, key_val_size, hidden_dim)

        out = some_multi_head_attn(query, key, val)
        correct_out = nn.functional.scaled_dot_product_attention(query, key, val)

        assert out.shape == correct_out.shape

    def test_attention_correct(self):
        """Edge case testing for correctness."""
        m_attn = MultiHeadAttention(4, 4, bias=False)

        batch_size = 3
        hidden_dim = 4
        query_size = 5
        key_val_size = 7

        query = torch.rand(batch_size, query_size, hidden_dim)
        key = torch.rand(batch_size, key_val_size, hidden_dim)
        torch.rand(batch_size, key_val_size, hidden_dim)

        # V = 0 -> attn score == 0
        out = m_attn(query, key, torch.zeros(batch_size, key_val_size, hidden_dim))
        assert torch.allclose(out, torch.zeros(out.shape))

    def test_gradient_flow(self, some_multi_head_attn):
        hidden_size = 32
        some_multi_head_attn.eval()  # Turn off dropout or other stochastic processes
        query = key = value = torch.rand((10, 5, hidden_size), requires_grad=True)
        output = some_multi_head_attn(
            value,
            key,
            query,
        )
        output.sum().backward()

        assert query.grad is not None, "Gradients not flowing to query"
        assert key.grad is not None, "Gradients not flowing to key"
        assert value.grad is not None, "Gradients not flowing to value"
