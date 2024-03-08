"""
`continuity.benchmarks.metrics.metric`

Metric base class.
"""

from abc import ABC, abstractmethod
from typing import Dict

from continuity.data import OperatorDataset
from continuity.operators import Operator


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, operator: Operator, dataset: OperatorDataset) -> Dict:
        """Evaluates the metric.

        Args:
            operator: operator for which the metric is evaluated.
            dataset: dataset on which the metric is evaluated.

        Returns:
            dict containing the results of the metric (keys "value" and "unit" should be in the dict).
        """

    def __str__(self):
        return self.name
