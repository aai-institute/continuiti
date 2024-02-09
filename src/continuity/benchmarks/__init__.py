"""
Benchmarks for operator learning.
"""

from abc import ABC, abstractmethod


class Benchmark(ABC):
    """Benchmark base class."""

    @abstractmethod
    def train_dataset(self):
        """Return training data set."""
        raise NotImplementedError

    @abstractmethod
    def test_dataset(self):
        """Return test data set."""
        raise NotImplementedError

    @abstractmethod
    def metric(self):
        """Return metric."""
        raise NotImplementedError
