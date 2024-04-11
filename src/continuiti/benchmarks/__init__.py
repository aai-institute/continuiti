"""
`continuiti.benchmarks`

Benchmarks for operator learning.
"""

from .benchmark import Benchmark
from .sine import SineRegular, SineUniform
from .flame import Flame
from .navierstokes import NavierStokes

__all__ = ["Benchmark", "SineRegular", "SineUniform", "Flame", "NavierStokes"]
