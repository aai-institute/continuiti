import torch
import torch.nn as nn

from continuity.networks import heterogeneous_normalized_attention


class TestHeterogeneousNormalizedAttention:
    def test_output_shape(self):
        batch_size = 3
        hidden_dim = 32
        query_size = 5
        key_val_size = 7

        query = torch.rand(batch_size, query_size, hidden_dim)
        key = torch.rand(batch_size, key_val_size, hidden_dim)
        val = torch.rand(batch_size, key_val_size, hidden_dim)

        out = heterogeneous_normalized_attention(query, key, val)
        correct_out = nn.functional.scaled_dot_product_attention(query, key, val)

        assert out.shape == correct_out.shape

    def test_attention_correct(self):
        """Edge case testing for correctness."""
        batch_size = 3
        hidden_dim = 4
        query_size = 5
        key_val_size = 7

        query = torch.rand(batch_size, query_size, hidden_dim)
        key = torch.rand(batch_size, key_val_size, hidden_dim)
        torch.rand(batch_size, key_val_size, hidden_dim)

        # V = 0 -> attn score == 0
        out = heterogeneous_normalized_attention(
            query, key, torch.zeros(batch_size, key_val_size, hidden_dim)
        )
        assert torch.allclose(out, torch.zeros(out.shape))

    def test_gradient_flow(self):
        hidden_size = 32
        query = key = value = torch.rand((10, 5, hidden_size), requires_grad=True)
        output = heterogeneous_normalized_attention(
            value,
            key,
            query,
        )
        output.sum().backward()

        assert query.grad is not None, "Gradients not flowing to query"
        assert key.grad is not None, "Gradients not flowing to key"
        assert value.grad is not None, "Gradients not flowing to value"
