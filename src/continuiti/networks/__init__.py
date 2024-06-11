"""
`continuiti.networks`

Networks in continuiti.
"""

from .fully_connected import FullyConnected
from .deep_residual_network import DeepResidualNetwork
from .attention import MultiHead
from .attention import ScaledDotProduct

__all__ = ["FullyConnected", "DeepResidualNetwork", "MultiHead", "ScaledDotProduct"]
