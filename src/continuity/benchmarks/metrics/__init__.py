"""
`continuity.benchmarks.metrics`

Metrics for benchmarks.
"""

from .error_metrics import L1Metric, MSEMetric
from .metric import Metric
from .operator_metrics import NumberOfParameters, SpeedOfEvaluation

__all__ = [
    "Metric",
    "L1Metric",
    "MSEMetric",
    "NumberOfParameters",
    "SpeedOfEvaluation",
]
