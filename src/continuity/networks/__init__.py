"""
`continuity.networks`

Networks in Continuity.
"""

from .fully_connected import FullyConnected
from .res_net import DeepResidualNetwork

__all__ = ["FullyConnected", "DeepResidualNetwork"]
