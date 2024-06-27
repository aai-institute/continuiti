"""
`continuiti.networks.multi_head_attention`

Multi-Head-Attention in continuiti.
"""

import torch
import torch.nn as nn

from .attention import Attention
from .scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(Attention):
    r"""Multi-Head Attention module.

    Module as described in the paper [Attention is All you
    Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
    with optional bias for the projections. This implementation allows to use
    attention implementations other than the standard scaled dot product
    attention implemented by the MultiheadAttention PyTorch module.

    $$MultiHead(Q,K,V)=Concat(head_1,\dots,head_n)W^O + b^O$$

    where

    $$head_i=Attention(QW_i^Q+b_i^Q, KW_i^K+b_i^K, VW_i^V+b_i^V).$$

    Args:
        hidden_dim: dimension of the hidden layers (embedding dimension).
        n_heads: number of attention heads.
        attention: implementation of attention (defaults to scaled dot product attention). Needs to have the arguments
            `query`, `key`, `value`, `attn_mask`, and `dropout_p`.
        dropout_p: dropout probability.
        bias: If True, then the projection onto the different heads is performed with bias.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        attention: Attention = None,
        dropout_p: float = 0,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout_p = dropout_p
        self.bias = bias

        if attention is None:
            attention = ScaledDotProductAttention()
        self.attention = attention

        self.head_dim = hidden_dim // n_heads
        assert (
            self.head_dim * n_heads == hidden_dim
        ), "hidden_dim must be divisible by n_heads"

        # projection networks
        self.query_project = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.key_project = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.value_project = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_project = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""Compute the attention scores.

        Args:
            query: Query tensor of shape (batch_size, target_sequence_length, hidden_dim).
            key: Key tensor of shape (batch_size, source_sequence_length, hidden_dim).
            value: Value tensor of shape (batch_size, source_sequence_length, hidden_dim).
            attn_mask: Attention mask of shape (batch_size, target_sequence_length, source_sequence_length).

        Returns:
            Attention scores of shape (batch_size, target_sequence_length, hidden_dim).
        """
        assert query.ndim == key.ndim == value.ndim == 3, (
            "Query, key, and value need to have three dimensions (batch_size, ..., hidden_dim). This format ensures that"
            "the module can correctly apply the multi-head attention mechanism, which includes splitting embeddings "
            "into multiple heads, applying the internal attention implementation for each head, concatenating and "
            "projecting results, while ensuring that the attention mask is applied correctly."
        )
        assert (
            query.size(0) == key.size(0) == value.size(0)
        ), "Batch size does not match for input tensors"
        assert (
            query.size(-1) == key.size(-1) == value.size(-1)
        ), "Embedding/hidden dimension does not match for input tensors"

        batch_size = query.size(0)
        src_len = key.size(1)
        tgt_len = query.size(1)

        # project values
        query = self.query_project(query)
        key = self.key_project(key)
        value = self.value_project(value)

        # form individual heads
        query = query.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # reshape attention mask to match heads
        if attn_mask is not None:
            assert (
                attn_mask.size(0) == batch_size
            ), "Attention mask batch size does not match input tensors."
            assert (
                attn_mask.size(1) == tgt_len
            ), "First dimension of the attention mask needs to match target length."
            assert (
                attn_mask.size(2) == src_len
            ), "Second dimension of the attention mask needs to match source length."

            attn_mask = attn_mask.unsqueeze(1)  # mask for a single head
            attn_mask = attn_mask.repeat(1, self.n_heads, 1, 1)  # mask for every head

        # perform attention
        attn_out = self.attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, self.hidden_dim)

        # output projection
        return self.out_project(attn_out)
