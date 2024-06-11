import pytest
import torch
import torch.nn as nn

from continuiti.networks import MultiHead, ScaledDotProduct


@pytest.fixture(scope="session")
def some_multi_head_attn():
    attn = ScaledDotProduct()
    return MultiHead(
        hidden_dim=32,
        n_heads=4,
        attention=attn,
        dropout_p=0.25,
        bias=True,
    )


class TestMultiHeadAttention:
    def test_can_initialize(self, some_multi_head_attn):
        assert isinstance(some_multi_head_attn, MultiHead)

    def test_output_shape(self, some_multi_head_attn):
        batch_size = 3
        query_size = 5
        key_val_size = 7

        query = torch.rand(batch_size, query_size, some_multi_head_attn.hidden_dim)
        key = torch.rand(batch_size, key_val_size, some_multi_head_attn.hidden_dim)
        val = torch.rand(batch_size, key_val_size, some_multi_head_attn.hidden_dim)

        out = some_multi_head_attn(query, key, val)

        gt_attn = nn.MultiheadAttention(
            embed_dim=some_multi_head_attn.hidden_dim,
            num_heads=some_multi_head_attn.n_heads,
            batch_first=True,
            bias=True,
        )
        correct_out, _ = gt_attn(query, key, val)

        assert out.shape == correct_out.shape

    def test_attention_correct(self):
        """Edge case testing for correctness."""
        m_attn = MultiHead(4, 4, bias=False)

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

    def test_equal_to_torch(self):
        heads = 2
        batch_size = 3
        target_length = 5
        source_length = 7
        embedding_dim = 8

        q = torch.rand(batch_size, target_length, embedding_dim)
        k = torch.rand(batch_size, source_length, embedding_dim)
        v = torch.rand(batch_size, source_length, embedding_dim)

        gt_attn = nn.MultiheadAttention(embedding_dim, heads, batch_first=True)
        attn = MultiHead(
            hidden_dim=embedding_dim,
            n_heads=heads,
            attention=nn.functional.scaled_dot_product_attention,
            dropout_p=0.0,
            bias=True,
        )

        # align in projection
        attn.key_project.weight = nn.Parameter(
            gt_attn.in_proj_weight[embedding_dim : 2 * embedding_dim, :]
        )
        attn.key_project.bias = nn.Parameter(
            gt_attn.in_proj_bias[embedding_dim : 2 * embedding_dim]
        )

        attn.value_project.weight = nn.Parameter(
            gt_attn.in_proj_weight[2 * embedding_dim :, :]
        )
        attn.value_project.bias = nn.Parameter(
            gt_attn.in_proj_bias[2 * embedding_dim :]
        )

        attn.query_project.weight = nn.Parameter(
            gt_attn.in_proj_weight[:embedding_dim, :]
        )
        attn.query_project.bias = nn.Parameter(gt_attn.in_proj_bias[:embedding_dim])

        # align out projection
        attn.out_project.weight = nn.Parameter(gt_attn.out_proj.weight)
        attn.out_project.bias = nn.Parameter(gt_attn.out_proj.bias)

        # forward pass
        out = attn(q, k, v)
        ground_truth, _ = gt_attn(q, k, v, need_weights=False)

        assert torch.allclose(out, ground_truth)
