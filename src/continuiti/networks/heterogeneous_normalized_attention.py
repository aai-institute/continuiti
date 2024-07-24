"""
`continuiti.networks.heterogeneous_normalized_attention`

Heterogeneous normalized attention block introduced by Hao et al. (https://proceedings.mlr.press/v202/hao23c).
"""

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from typing import Optional

from .attention import UniformMaskAttention


class HeterogeneousNormalizedAttention(UniformMaskAttention):
    r"""Heterogeneous normalized attention with uniform masks.

    Computes the normalization coefficient alpha for attention mechanisms, as proposed by Hao et al. in "GNOT: A
    General Neural Operator Transformer for Operator Learning" (https://proceedings.mlr.press/v202/hao23c). The
    attention score is calculated by normalizing the keys and queries
    $$\tilde{q}_i = Softmax(\frac{\exp(q_{i,j})}{\sum_j\exp(q_{i,j})}$$,
    $$\tilde{k}_i = Softmax(\frac{\exp(k_{i,j})}{\sum_j\exp(k_{i,j})}$$, and then calculating the attention without
    softmax using $$z_t=\sum_i \frac{\tilde{q}_t \cdot \tilde{k}_i}{\sum_j \tilde{q}_t \cdot \tilde{k}_j}\cdot v_i$$.
    The computational cost for this is O((M+N)n_e^2) (M=number of keys/values, N=number of queries, n_e=embedding_dim),
    now is linear with respect to the sequence length.

    Args:
        tau: Temperature parameter controlling the sharpness of the softmax operation.
    """

    def __init__(self, tau: float = 1.0, dropout_p: float = 0.0):
        super().__init__()
        self.tau = tau
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass.
        Args:
            query: Tensor of shape (batch_size, ..., d_q, embedding_dim).
            key: Tensor of shape (batch_size, ..., d_kv, embedding_dim).
            value: Tensor of shape (batch_size, ..., d_kv, embedding_dim).
            attn_mask: Attention mask of shape (batch_size, ..., d_kv). A boolean mask where a True indicates that
                a value should be taken into consideration in the calculations.
        Returns:
            Attention output of shape (batch_size, ..., d_q, e_dim).
        """
        assert (
            query.ndim == key.ndim
        ), "Number of dimensions in queries and keys should match."
        assert (
            query.ndim == value.ndim
        ), "Number of dimensions in queries and values should match."

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(-1)
            key = torch.masked_fill(key, attn_mask.logical_not(), float("-inf"))
            value = torch.masked_fill(value, attn_mask.logical_not(), 0.0)

        q_tilde = softmax(query, dim=-1)
        if attn_mask is not None:
            q_tilde = torch.nan_to_num(
                q_tilde, nan=0.0
            )  # masking might ignore queries entirely resulting in nan in softmax

        k_tilde = softmax(key / self.tau, dim=-1)
        if attn_mask is not None:
            k_tilde = torch.nan_to_num(
                k_tilde, nan=0.0
            )  # masking might ignore keys entirely resulting in nan in softmax

        alpha = torch.matmul(q_tilde, k_tilde.transpose(-1, -2))
        alpha = torch.sum(alpha, dim=-1, keepdim=True)
        if attn_mask is not None:
            alpha[alpha == 0.0] = 1.0  # numerical stability

        mat = k_tilde * value
        mat = self.dropout(mat)
        mat = torch.sum(mat, dim=-2, keepdim=True)

        return q_tilde * mat / alpha
