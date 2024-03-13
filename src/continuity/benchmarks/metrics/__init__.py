"""
`continuity.benchmarks.metrics`

Metrics for benchmarks.
"""

from .error_metrics import L1Metric, MSEMetric
from .metric import Metric
from .operator_metrics import NumberOfParametersMetric, SpeedOfEvaluationMetric

__all__ = [
    "Metric",
    "L1Metric",
    "MSEMetric",
    "NumberOfParametersMetric",
    "SpeedOfEvaluationMetric",
]
