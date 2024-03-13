import torch

from continuity.benchmarks import Benchmark
from continuity.benchmarks.sine import (
    create_sine_dataset,
    create_sine_benchmark,
    sine_benchmark,
)
from continuity.data import OperatorDataset


def test_return_type_correct():
    assert isinstance(create_sine_dataset(2, 3, 5), OperatorDataset)
    assert isinstance(create_sine_benchmark(1, 2, 3, 5), Benchmark)


def test_can_initialize_default():
    assert isinstance(sine_benchmark, Benchmark)


def test_default_is_identity():
    for dataset in [sine_benchmark.train_dataset, sine_benchmark.test_dataset]:
        for x, u, y, v in dataset:
            assert torch.allclose(x, y)
            assert torch.allclose(u, v)
