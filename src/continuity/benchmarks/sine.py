"""Sine benchmark."""

from continuity.benchmarks import Benchmark
from continuity.data import split
from continuity.data.datasets import Sine
from continuity.operators.losses import Loss, MSELoss
from torch.utils.data import Dataset


class SineBenchmark(Benchmark):
    """Sine benchmark."""

    def __init__(self):
        self.num_sensors = 32
        self.size = 100
        self.batch_size = 1

        self.dataset = Sine(
            num_sensors=32,
            size=100,
        )

        self.train_dataset, self.test_dataset = split(self.dataset, 0.9)

    def dataset(self) -> Dataset:
        """Return data set."""
        return self.dataset

    def train_dataset(self) -> Dataset:
        """Return training data set."""
        return self.train_dataset

    def test_dataset(self) -> Dataset:
        """Return test data set."""
        return self.test_dataset

    def metric(self) -> Loss:
        """Return metric."""
        return MSELoss()
