import optuna
import torch
from continuity.benchmarks.run import BenchmarkRunner, RunConfig
from continuity.benchmarks import SineRegular, SineUniform
from continuity.operators import (
    DeepONet,
    BelNet,
    FourierNeuralOperator,
    DeepNeuralOperator,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
n_trials = 10


def run_single(benchmark_factory, operator_factory):
    """Run hyper-parameter sweep for one benchmark and operator."""

    def objective(trial):
        seed = trial.suggest_int("seed", 0, 100)

        config = RunConfig(
            benchmark_factory,
            operator_factory,
            seed=seed,
            device=device,
        )

        test_loss = BenchmarkRunner.run(config, trial.params)
        return test_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)


def run_all():
    """Run benchmarks for all operators."""

    # Benchmarks
    benchmarks = [
        SineRegular,
        SineUniform,
    ]

    # Operators
    operators = [
        DeepONet,
        FourierNeuralOperator,
        BelNet,
        DeepNeuralOperator,
    ]

    # Run all combinations
    for benchmark_factory in benchmarks:
        for operator_factory in operators:
            run_single(benchmark_factory, operator_factory)


if __name__ == "__main__":
    run_all()
