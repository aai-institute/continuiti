"""
`continuity.benchmarks.benchmark`

Benchmark base class.
"""

from typing import List
from continuity.data import OperatorDataset
from continuity.benchmarks.metrics import Metric


class Benchmark:
    """Benchmark base class."""

    def __init__(
        self,
        train_dataset: OperatorDataset,
        test_dataset: OperatorDataset,
        metrics: List[Metric],
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.metrics = metrics
