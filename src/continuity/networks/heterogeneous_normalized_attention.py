import torch
import torch.nn.functional as F


def heterogeneous_normalized_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    tau: float = 1.0,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    r"""Heterogeneous normalized attention.

    Computes the normalization coefficient alpha for attention mechanisms, as proposed by Hao et al. in "GNOT: A General
     Neural Operator Transformer for Operator Learning" (https://proceedings.mlr.press/v202/hao23c).

    The attention score is calculated by normalizing the keys and queries
    $$\tilde{q}_i = Softmax(\frac{\exp(q_{i,j})}{\sum_j\exp(q_{i,j})}$$,
    $$\tilde{k}_i = Softmax(\frac{\exp(k_{i,j})}{\sum_j\exp(k_{i,j})}$$,
    and then calculating the attention without softmax using
    $$z_t=\sum_i \frac{\tilde{q}_t \cdot \tilde{k}_i}{\sum_j \tilde{q}_t \cdot \tilde{k}_j}\cdot v_i$$.

    The computational cost for this is O((M+N)n_e^2) (M=number of keys/values, N=number of queries, n_e=embedding_dim),
    now is linear with respect to the sequence length.

    Args:
        query: Tensor of shape (batch_size, d_q, e_dim).
        key: Tensor of shape (batch_size, d_kv, e_dim).
        value: Tensor of shape (batch_size, d_kv, e_dim).
        attn_mask:
        tau: Temperature parameter controlling the sharpness of the softmax operation

    Returns:
        Attention output of shape (batch_size, d_q, e_dim).
    """
    if attn_mask is not None:
        query = query.masked_fill(attn_mask.logical_not(), float("-inf"))
        key = key.masked_fill(attn_mask.logical_not(), float("-inf"))

    query_norm = F.softmax(query / tau, dim=-1)
    key_norm = F.softmax(key / tau, dim=-1)

    dot_prod = torch.matmul(query_norm, key_norm.transpose(-2, -1))
    alpha_inv = torch.sum(dot_prod, dim=-1, keepdim=True)

    attn_weights = dot_prod / alpha_inv
    attn_out = torch.matmul(attn_weights, value)

    return attn_out
