import random
import torch
import mlflow
import numpy as np
from datetime import datetime
from typing import Optional
from continuiti.benchmarks.run import RunConfig
from continuiti.trainer import Trainer
from continuiti.trainer.callbacks import MLFlowLogger
from continuiti.trainer.device import get_device


class BenchmarkRunner:
    """Benchmark runner."""

    @staticmethod
    def run(config: RunConfig, params_dict: Optional[dict] = None) -> float:
        """Run a benchmark.

        Args:
            config: run configuration.
            params_dict: dictionary of parameters to log.

        Returns:
            Test loss.

        """
        # Device
        device = torch.device(config.device) or get_device()

        # Rank
        rank = device.index or 0

        # Benchmark
        benchmark = config.benchmark_factory()

        # Operator
        shapes = benchmark.train_dataset.shapes
        operator = config.operator_factory(shapes, device=device)

        # Log parameters
        if rank == 0:
            if params_dict is None:
                params_dict = {}

            param_str = " ".join(f"{key}={value}" for key, value in params_dict.items())

            # MLFLow
            mlflow.set_experiment(f"{benchmark}")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run_name = f"{operator} {timestamp}"
            tags = {
                "benchmark": str(benchmark),
                "operator": str(operator),
                "device": str(config.device),
            }
            mlflow.start_run(run_name=run_name, tags=tags)

            # Log parameters
            if params_dict is not None:
                for key, value in params_dict.items():
                    mlflow.log_param(key, value)

            if "seed" not in params_dict:
                mlflow.log_param("seed", config.seed)
            if "lr" not in params_dict:
                mlflow.log_param("lr", config.lr)
            if "batch_size" not in params_dict:
                mlflow.log_param("batch_size", config.batch_size)
            if "tol" not in params_dict:
                mlflow.log_param("tol", config.tol)
            if "max_epochs" not in params_dict:
                mlflow.log_param("max_epochs", config.max_epochs)
            mlflow.log_metric("num_params", operator.num_params())

        # Seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # For now, take the sum of all losses in benchmark
        def loss_fn(*args):
            return sum(loss(*args) for loss in benchmark.losses)

        # Trainer
        optimizer = torch.optim.Adam(operator.parameters(), lr=config.lr)
        trainer = Trainer(
            operator,
            optimizer,
            loss_fn=loss_fn,
            device=config.device,
            verbose=config.verbose,
        )

        if rank == 0:
            print(f"> {benchmark} {operator} {param_str}")

        # Train
        callbacks = None
        if rank == 0:
            callbacks = [MLFlowLogger(operator)]

        logs = trainer.fit(
            benchmark.train_dataset,
            tol=config.tol,
            epochs=config.max_epochs,
            callbacks=callbacks,
            batch_size=config.batch_size,
            test_dataset=benchmark.test_dataset,
        )

        # Return test loss
        return logs.loss_test
