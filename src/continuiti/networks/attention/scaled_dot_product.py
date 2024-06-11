"""
`continuiti.networks.attention.scaled_dot_product`

Scaled dot product attention module.
"""
import torch

from .attention import Attention
from torch.nn.functional import scaled_dot_product_attention


class ScaledDotProduct(Attention):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch,
                attn_mask: torch.Tensor = None, dropout_p: float = 0.) -> torch.Tensor:
        return scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )
