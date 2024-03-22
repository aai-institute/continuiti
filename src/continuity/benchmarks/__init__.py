"""
`continuity.benchmarks`

Benchmarks for operator learning.
"""

from .benchmark import Benchmark
from .sine import SineRegular, SineUniform
from .flame import Flame

__all__ = ["Benchmark", "SineRegular", "SineUniform", "Flame"]
