"""
`continuiti.discrete`

Functionalities handling discretization of continuous functionals.
"""

from .uniform import UniformBoxSampler
from .regular_grid import RegularGridSampler

__all__ = ["UniformBoxSampler", "RegularGridSampler"]
