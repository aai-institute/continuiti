import random
import torch
import numpy as np
from time import time
from typing import Callable, Union
from dataclasses import dataclass

from continuity.benchmarks import Benchmark
from continuity.operators import Operator, OperatorShapes
from continuity.trainer import Trainer
from continuity.data.utility import dataset_loss


@dataclass
class RunConfig:
    benchmark_name: str
    benchmark_factory: Callable[[], Benchmark]
    operator_name: str
    operator_factory: Callable[[OperatorShapes], Operator]
    seed: int = 0
    lr: float = 1e-3
    tol: float = 1e-5
    max_epochs: int = 100
    device: Union[torch.device, str] = "cpu"


class BenchmarkRunner:
    def run(self, run: RunConfig) -> dict:
        print(f"[{run.seed}] {run.benchmark_name} {run.operator_name}")

        bm = run.benchmark_factory()
        shapes = bm.train_dataset.shapes
        op = run.operator_factory(shapes)

        random.seed(run.seed)
        np.random.seed(run.seed)
        torch.manual_seed(run.seed)

        # For now, take the sum of all losses in benchmark
        def loss_fn(*args):
            return sum(loss(*args) for loss in bm.losses)

        trainer = Trainer(op, lr=run.lr, loss_fn=loss_fn, device=run.device)

        start = time()
        stats = trainer.fit(bm.train_dataset, tol=run.tol, epochs=run.max_epochs)
        end = time()
        stats["time/train"] = start - end

        start = time()
        loss_test = dataset_loss(bm.test_dataset, op, loss_fn)
        end = time()
        stats["loss/test"] = loss_test
        stats["time/test"] = start - end

        # Append run configuration
        stats["Benchmark"] = run.benchmark_name
        stats["Operator"] = run.operator_name
        stats["num_params"] = op.num_params()
        stats["seed"] = str(run.seed)
        stats["lr"] = str(run.lr)
        stats["tol"] = str(run.tol)
        stats["max_epochs"] = str(run.max_epochs)

        return stats
