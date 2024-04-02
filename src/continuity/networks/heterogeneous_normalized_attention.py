import torch
import torch.nn.functional as F


def heterogeneous_normalized_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """

    Args:
        query:
        key:
        value:
        attn_mask:
        tau: Factor controlling the sharpness of the softmax normalization.

    Returns:

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
