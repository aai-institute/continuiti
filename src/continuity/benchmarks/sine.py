"""
`continuity.benchmarks.sine`

Sine benchmark.
"""

from continuity.benchmarks import Benchmark
from continuity.data import split
from continuity.data.sine import Sine
from continuity.benchmarks.metrics import MSEMetric


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

        train_dataset, test_dataset = split(self.dataset, 0.9)
        super().__init__(train_dataset, test_dataset, [MSEMetric()])
