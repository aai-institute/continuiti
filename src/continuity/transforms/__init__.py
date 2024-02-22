from typing import TypeAlias

from .transform import Transform
from .compose import Compose
from .scaling import ZNormalization

Standardization: TypeAlias = ZNormalization

__all__ = [
    "Transform",
    "Compose",
    "ZNormalization",
    "Standardization",
]
