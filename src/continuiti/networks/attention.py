"""
`continuiti.networks.attention`

Attention base class in continuiti.
"""

from abc import abstractmethod
import torch.nn as nn
import torch
from typing import Optional


class UniformMaskAttention(nn.Module):
    """Base class for various attention implementations with uniform masking.

    Attention assigns different parts of an input varying importance without set kernels. The importance of different
    components is designated using "soft" weights. These weights are assigned according to specific algorithms (e.g.
    scaled-dot-product attention).
    Uniform masking refers to the characteristic that all queries use the same mask. This restriction allows to remove
    the query dimension from the mask. All queries have access to the same key/value pairs.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculates the attention scores.

        Args:
            query: query tensor; shape (batch_size, target_seq_length, hidden_dim).
            key: key tensor; shape (batch_size, source_seq_length, hidden_dim).
            value: value tensor; shape (batch_size, source_seq_length, hidden_dim).
            attn_mask: tensor indicating which values are used to calculate the output;
                shape (batch_size, source_seq_length).

        Returns:
            tensor containing the outputs of the attention implementation;
                shape (batch_size, target_seq_length, hidden_dim).
        """
