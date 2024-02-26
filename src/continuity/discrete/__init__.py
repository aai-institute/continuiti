"""
`continuity.discrete`

Functionalities handling discretization of continuous functionals.
"""

from .uniform import UniformBoxSampler
from .uniform_grid import UniformGridSampler

__all__ = ["UniformBoxSampler", "UniformGridSampler"]
