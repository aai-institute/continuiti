"""Sine benchmark."""

from continuity.benchmarks import Benchmark
from continuity.data import DataSet, split
from continuity.data.datasets import Sine
from continuity.operators.losses import Loss, MSELoss


class SineBenchmark(Benchmark):
    """Sine benchmark."""

    def __init__(self):
        self.num_sensors = 32
        self.size = 100
        self.batch_size = 1

        self.dataset = Sine(
            num_sensors=32,
            size=100,
            batch_size=1,
        )

        self.train_dataset, self.test_dataset = split(self.dataset, 0.9)

    def dataset(self) -> DataSet:
        """Return data set."""
        return self.dataset

    def train_dataset(self) -> DataSet:
        """Return training data set."""
        return self.train_dataset

    def test_dataset(self) -> DataSet:
        """Return test data set."""
        return self.test_dataset

    def metric(self) -> Loss:
        """Return metric."""
        return MSELoss()
