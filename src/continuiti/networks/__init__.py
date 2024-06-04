"""
`continuiti.networks`

Networks in continuiti.
"""

from .fully_connected import FullyConnected
from .deep_residual_network import DeepResidualNetwork
from .attention import MultiHeadAttention

__all__ = ["FullyConnected", "DeepResidualNetwork", "MultiHeadAttention"]
