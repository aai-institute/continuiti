import pytest
import torch
import torch.nn as nn

from continuiti.networks import MultiHeadAttention, ScaledDotProductAttention


@pytest.fixture(scope="session")
def some_multi_head_attn():
    return MultiHeadAttention(
        hidden_dim=32,
        n_heads=4,
        attention=ScaledDotProductAttention(dropout_p=0.25),
        bias=True,
    )


@pytest.fixture(scope="class")
def random_qkv():
    batch_size = 3
    target_length = 5
    source_length = 7
    embedding_dim = 8

    q = torch.rand(batch_size, target_length, embedding_dim)
    k = torch.rand(batch_size, source_length, embedding_dim)
    v = torch.rand(batch_size, source_length, embedding_dim)
    return q, k, v


class TestMultiHeadAttention:
    def test_can_initialize(self, some_multi_head_attn):
        assert isinstance(some_multi_head_attn, MultiHeadAttention)

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

    def test_zero_value(self, random_qkv):
        """Edge case testing for correctness."""
        q, k, v = random_qkv
        v = torch.zeros(v.shape)

        m_attn = MultiHeadAttention(q.size(-1), 4, bias=False)

        # V = 0 -> attn score == 0
        out = m_attn(q, k, v)
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

    def test_equal_to_torch(self, random_qkv):
        q, k, v = random_qkv
        mask = torch.rand(q.size(0), q.size(1), k.size(1)) < 0.2

        heads = 2
        embedding_dim = q.size(-1)

        gt_attn = nn.MultiheadAttention(q.size(-1), heads, batch_first=True)
        attn = MultiHeadAttention(
            hidden_dim=q.size(-1),
            n_heads=heads,
            attention=ScaledDotProductAttention(dropout_p=0.0),
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
        out = attn(q, k, v, attn_mask=mask)

        # torch applies masks differently to scaled-dot-product and multi-head attention (inversed).
        gt_mask = torch.repeat_interleave(mask, heads, 0).logical_not()
        ground_truth, _ = gt_attn(q, k, v, need_weights=False, attn_mask=gt_mask)

        assert torch.allclose(
            out[~torch.isnan(out)], ground_truth[~torch.isnan(ground_truth)]
        )

    def test_full_mask_identical_to_none(self, random_qkv):
        heads = 2
        q, k, v = random_qkv

        mask = torch.ones(q.size(0), q.size(1), k.size(1))

        attn = MultiHeadAttention(
            hidden_dim=q.size(-1),
            n_heads=heads,
            attention=ScaledDotProductAttention(dropout_p=0.0),
            bias=True,
        )

        # forward pass
        out_masked = attn(q, k, v, attn_mask=mask)
        out_none = attn(q, k, v)

        assert torch.allclose(out_masked, out_none)

    def test_mask_all_but_one(self, random_qkv):
        q, k, v = random_qkv
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True

        # Masks out the last kvs
        mask = torch.ones(q.size(0), q.size(1), k.size(1), dtype=torch.bool)
        mask[:, :, -1] = 0

        attn = MultiHeadAttention(
            hidden_dim=q.size(-1),
            n_heads=2,
            attention=ScaledDotProductAttention(dropout_p=0.0),
            bias=True,
        )
        out = attn(q, k, v, attn_mask=mask)

        eq = torch.sum(out)
        eq.backward()

        assert not torch.any(torch.isnan(q.grad))
        assert not torch.any(
            torch.isclose(q.grad, torch.zeros(q.shape))
        )  # all queries have a non-zero gradient

        assert not torch.any(torch.isnan(v.grad))
        unmasked_rows = v.grad[:, :-1, :]  # gradient on unmasked values is non-zero
        assert not torch.any(
            torch.isclose(unmasked_rows, torch.zeros(unmasked_rows.shape))
        )
        masked_row = v.grad[:, -1, :]  # gradient on masked value is zero
        assert torch.allclose(masked_row, torch.zeros(masked_row.shape))

        assert not torch.any(torch.isnan(k.grad))
        unmasked_rows = k.grad[:, :-1, :]  # gradient on unmasked keys is non-zero
        assert not torch.any(
            torch.isclose(unmasked_rows, torch.zeros(unmasked_rows.shape))
        )
        masked_row = k.grad[:, -1, :]  # gradient on masked key is zero
        assert torch.allclose(masked_row, torch.zeros(masked_row.shape))
