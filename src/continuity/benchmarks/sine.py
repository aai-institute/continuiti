"""
`continuity.benchmarks.sine`

Sine benchmark.
"""

from continuity.benchmarks import Benchmark
from continuity.data import split
from continuity.data.sine import Sine
from continuity.operators.losses import Loss, MSELoss
from torch.utils.data import Dataset


class SineBenchmark(Benchmark):
    """Sine benchmark.

    A sine wave data set containing 100 samples with 32 sensors,
    split randomly into 90% training and 10% test data.
    """

    def __init__(self):
        self.num_sensors = 32
        self.size = 100

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
        """Return MSELoss."""
        return MSELoss()
