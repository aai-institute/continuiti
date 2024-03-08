"""
`continuity.benchmarks.sine`

Sine benchmark.
"""

from continuity.benchmarks import Benchmark
from continuity.data.sine import Sine
from continuity.benchmarks.metrics import (
    MSEMetric,
    L1Metric,
    SpeedOfEvaluationMetric,
    NumberOfParametersMetric,
)


class SineBenchmark(Benchmark):
    """Sine benchmark.

    A sine wave data set containing 100 samples with 32 sensors,
    split randomly into 90% training and 10% test data.
    """

    def __init__(self):
        train_dataset = Sine(num_sensors=32, size=90)
        test_dataset = Sine(num_sensors=32, size=10)

        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            metrics=[
                MSEMetric(),
                L1Metric(),
                NumberOfParametersMetric(),
                SpeedOfEvaluationMetric(),
            ],
        )
