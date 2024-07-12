"""
`continuiti.operators.modulus`

Operators from NVIDIA Modulus wrapped in continuiti.
"""

# Test if we can import NVIDIA Modulus
try:
    import modulus  # noqa: F40
except ImportError:
    raise ImportError("NVIDIA Modulus not found!")

from .fno import FNO

__all__ = [
    "FNO",
]
