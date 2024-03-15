import torch

from continuity.benchmarks import Benchmark
from continuity.benchmarks.sine import SineBenchmark
from continuity.data import OperatorDataset


def test_return_type_correct():
    sine_benchmark = SineBenchmark()
    assert isinstance(sine_benchmark.train_dataset, OperatorDataset)
    assert isinstance(sine_benchmark.test_dataset, OperatorDataset)


def test_can_initialize_default():
    assert isinstance(SineBenchmark(), Benchmark)


def test_default_is_identity():
    sine_benchmark = SineBenchmark()
    for dataset in [sine_benchmark.train_dataset, sine_benchmark.test_dataset]:
        for x, u, y, v in dataset:
            assert torch.allclose(x, y)
            assert torch.allclose(u, v)
