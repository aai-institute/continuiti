import random
import torch
import numpy as np
from time import time
from typing import Callable, Union
from dataclasses import dataclass

from continuity.benchmarks import Benchmark
from continuity.operators import Operator, OperatorShapes
from continuity.trainer import Trainer
from continuity.trainer.callbacks import (
    PrintTrainingLoss,
    LossHistory,
    LinearLRScheduler,
)
from continuity.data.utility import dataset_loss


@dataclass
class RunConfig:
    benchmark_name: str
    benchmark_factory: Callable[[], Benchmark]
    operator_name: str
    operator_factory: Callable[[OperatorShapes], Operator]
    seed: int = 0
    lr: float = 1e-3
    tol: float = 0
    max_epochs: int = 1000
    device: Union[torch.device, str] = "cpu"


class BenchmarkRunner:
    def run(self, run: RunConfig) -> dict:
        print(f"{run.benchmark_name} {run.operator_name} seed={run.seed}")

        bm = run.benchmark_factory()
        shapes = bm.train_dataset.shapes
        op = run.operator_factory(shapes)

        random.seed(run.seed)
        np.random.seed(run.seed)
        torch.manual_seed(run.seed)

        # For now, take the sum of all losses in benchmark
        def loss_fn(*args):
            return sum(loss(*args) for loss in bm.losses)

        optimizer = torch.optim.Adam(op.parameters(), lr=run.lr)
        trainer = Trainer(op, optimizer, loss_fn=loss_fn, device=run.device)
        max_epochs = int(run.max_epochs)

        history = LossHistory()
        lr_scheduler = LinearLRScheduler(optimizer, max_epochs)
        callbacks = [PrintTrainingLoss(), history, lr_scheduler]

        start = time()
        stats = trainer.fit(
            bm.train_dataset,
            tol=run.tol,
            epochs=max_epochs,
            callbacks=callbacks,
            val_dataset=bm.test_dataset,
        )

        end = time()
        stats["time/train"] = end - start

        start = time()
        loss_test = dataset_loss(bm.test_dataset, op, loss_fn, run.device)
        end = time()
        stats["loss/test"] = loss_test
        stats["time/test"] = end - start

        # Append run configuration
        stats["Benchmark"] = run.benchmark_name
        stats["Operator"] = run.operator_name
        stats["params"] = op.num_params()
        stats["seed"] = str(run.seed)
        stats["tol"] = str(run.tol)
        stats["max_epochs"] = str(run.max_epochs)
        stats["train_history"] = history.train_history

        return stats
