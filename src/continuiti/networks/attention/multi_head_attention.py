"""
`continuiti.networks.multi_head_attention`

Multi-Head-Attention in continuiti.
"""

from typing import Callable
import torch
import torch.nn as nn

from .attention import Attention


class MultiHeadAttention(Attention):
    r"""Multi-Head Attention module.

    Module as described in the paper [Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
    with optional bias for the projections.

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
            attention: Callable = nn.functional.scaled_dot_product_attention,
            dropout_p: float = 0,
            bias: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.attention = attention
        self.dropout_p = dropout_p
        self.bias = bias

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
            value: torch,
            attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        assert query.ndim == 3
        assert key.ndim == 3
        assert value.ndim == 3

        batch_size = query.size(0)

        # project values
        query = self.query_project(query)
        key = self.key_project(key)
        value = self.value_project(value)

        # form individual heads
        query = query.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # perform attention
        attn_out = self.attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, self.hidden_dim)

        # output projection
        return self.out_project(attn_out)
