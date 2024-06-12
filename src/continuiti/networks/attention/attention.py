"""
`continuiti.networks.attention.attention`

Attention base class in continuiti.
"""

from abc import abstractmethod
import torch.nn as nn
import torch


class Attention(nn.Module):
    """Base class for various attention implementations.

    Attention assigns different parts of an input varying importance without set kernels. The importance of different
    components is designated using "soft" weights. These weights are assigned according to specific algorithms (e.g.
    scaled-dot-product attention).

    Args:
        dropout_p: dropout probability.
    """

    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout_p = dropout_p

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Calculates the attention scores.

        Args:
            query: query tensor; shape (batch_size, target_seq_length, hidden_dim)
            key: key tensor; shape (batch_size, source_seq_length, hidden_dim)
            value: value tensor; shape (batch_size, source_seq_length, hidden_dim)
            attn_mask: tensor indicating which values are used to calculate the output;
                shape (batch_size, target_seq_length, source_seq_length)

        Returns:
            tensor containing the outputs of the attention implementation;
                shape (batch_size, target_seq_length, hidden_dim)
        """
