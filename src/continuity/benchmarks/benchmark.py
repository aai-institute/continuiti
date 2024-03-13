"""
`continuity.benchmarks.benchmark`

Benchmark base class.
"""

from typing import List
from dataclasses import dataclass, field
from continuity.data import OperatorDataset
from continuity.benchmarks.metrics import (
    Metric,
    L1Metric,
    MSEMetric,
    NumberOfParametersMetric,
    SpeedOfEvaluationMetric,
)


@dataclass
class Benchmark:
    """Benchmark class."""

    train_dataset: OperatorDataset
    test_dataset: OperatorDataset
    metrics: List[Metric] = field(
        default_factory=lambda: [
            L1Metric(),
            MSEMetric(),
            NumberOfParametersMetric(),
            SpeedOfEvaluationMetric(),
        ]
    )
