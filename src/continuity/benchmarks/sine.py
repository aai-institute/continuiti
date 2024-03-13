"""
`continuity.benchmarks.sine`

Sine benchmark.
"""

from continuity.benchmarks import Benchmark
from continuity.data.sine import Sine

"""Sine benchmark.

A sine wave data set containing 100 samples with 32 sensors,
split randomly into 90% training and 10% test data.
"""
sine_benchmark = Benchmark(
    train_dataset=Sine(num_sensors=32, size=90),
    test_dataset=Sine(num_sensors=32, size=10),
)
