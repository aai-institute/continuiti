"""
`continuiti.networks.scaled_dot_product_attention`

Scaled dot product attention module.
"""
import torch
from typing import Optional

from .attention import UniformMaskAttention
from torch.nn.functional import scaled_dot_product_attention


class ScaledDotProductAttention(UniformMaskAttention):
    """Scaled dot product attention module with uniform mask.

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
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate attention scores.

        Args:
            query: query tensor; shape (batch_size, target_seq_length, hidden_dim).
            key: key tensor; shape (batch_size, source_seq_length, hidden_dim).
            value: value tensor; shape (batch_size, source_seq_length, hidden_dim).
            attn_mask: tensor indicating which values are used to calculate the output; shape
                (batch_size, source_seq_length). Defaults to None.

        Returns:
            tensor containing the outputs of the attention implementation; shape
                (batch_size, target_seq_length, hidden_dim).
        """
        dropout_p = self.dropout_p if self.training else 0.0

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)

        return scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
