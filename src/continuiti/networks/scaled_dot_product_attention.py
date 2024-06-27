"""
`continuiti.networks.scaled_dot_product_attention`

Scaled dot product attention module.
"""
import torch

from .attention import Attention
from torch.nn.functional import scaled_dot_product_attention


class ScaledDotProductAttention(Attention):
    """Scaled dot product attention module.

    This module is a wrapper for the torch implementation of the scaled dot
    product attention mechanism as described in the paper "Attention Is All You
    Need" by Vaswani et al. (2017). This attention mechanism computes the
    attention weights based on the dot product of the query and key matrices,
    scaled by the square root of the dimension of the key vectors. The weights
    are then applied to the value vectors to obtain the final output.
    """

    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        dropout_p = self.dropout_p if self.training else 0.0
        return scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
