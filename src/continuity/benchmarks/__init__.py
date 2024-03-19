"""
`continuity.benchmarks`

Benchmarks for operator learning.
"""

from .benchmark import Benchmark
from .sine import SineRegular, SineUniform

__all__ = ["Benchmark", "SineRegular", "SineUniform"]
