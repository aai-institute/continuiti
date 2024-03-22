import torch
from typing import Callable, Union
from dataclasses import dataclass

from continuity.benchmarks import Benchmark
from continuity.operators import Operator, OperatorShapes
from continuity.trainer.device import get_device


@dataclass
class RunConfig:
    """Run configuration.

    Args:
        benchmark_factory: Benchmark factory. Callable that returns a Benchmark.
        operator_factory: Operator factory. Callable that takes OperatorShapes and returns an Operator.
        seed: Random seed.
        lr: Learning rate.
        tol: Threshold for stopping criterion.
        max_epochs: Maximum number of epochs.
        batch_size: Batch size.
        device: Device.
    """

    benchmark_factory: Callable[[], Benchmark]
    operator_factory: Callable[[OperatorShapes], Operator]
    seed: int = 0
    lr: float = 1e-3
    tol: float = 0
    max_epochs: int = 1000
    batch_size: int = 32
    device: Union[torch.device, str] = get_device()
