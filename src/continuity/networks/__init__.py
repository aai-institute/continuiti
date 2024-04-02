"""
`continuity.networks`

Networks in Continuity.
"""

from .fully_connected import FullyConnected
from .res_net import DeepResidualNetwork
from .multi_head_attention import MultiHeadAttention
from .function_encoder import FunctionEncoderLayer, FunctionEncoder

__all__ = [
    "FullyConnected",
    "DeepResidualNetwork",
    "MultiHeadAttention",
    "FunctionEncoderLayer",
    "FunctionEncoder",
]
