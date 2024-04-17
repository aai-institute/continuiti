import optuna
import torch
from functools import partial
from multiprocessing import Pool
from continuiti.benchmarks.run import BenchmarkRunner, RunConfig
from continuiti.benchmarks import SineRegular, SineUniform
from continuiti.operators import (
    DeepONet,
    BelNet,
    FourierNeuralOperator,
    DeepNeuralOperator,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
n_trials = 10
n_processes = 12


def run_single(benchmark_factory, operator_factory):
    """Run hyper-parameter sweep for one benchmark and operator."""

    def objective(trial):
        seed = trial.suggest_int("seed", 0, 100)

        config = RunConfig(
            benchmark_factory,
            operator_factory(trial),
            seed=seed,
            device=device,
        )

        test_loss = BenchmarkRunner.run(config, trial.params)
        return test_loss

    # Run random search to not fit the test set
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=1)


def DeepONetFactory(trial):
    branch_width = trial.suggest_categorical("branch_width", [8, 32])
    branch_depth = trial.suggest_categorical("branch_depth", [1, 8])
    trunk_width = trial.suggest_categorical("trunk_width", [8, 32])
    trunk_depth = trial.suggest_categorical("trunk_depth", [1, 8])
    basis_functions = trial.suggest_categorical("basis_functions", [8, 32])
    return partial(
        DeepONet,
        branch_width=branch_width,
        branch_depth=branch_depth,
        trunk_width=trunk_width,
        trunk_depth=trunk_depth,
        basis_functions=basis_functions,
    )


def FourierNeuralOperatorFactory(trial):
    width = trial.suggest_categorical("width", [1, 4, 8])
    depth = trial.suggest_categorical("depth", [1, 2, 3])
    return partial(FourierNeuralOperator, width=width, depth=depth)


def BelNetFactory(trial):
    K = trial.suggest_categorical("K", [8, 16])
    N_1 = trial.suggest_categorical("N_1", [8, 16])
    D_1 = trial.suggest_categorical("D_1", [4, 8])
    N_2 = trial.suggest_categorical("N_2", [8, 16])
    D_2 = trial.suggest_categorical("D_2", [4, 8])
    return partial(BelNet, K=K, N_1=N_1, D_1=D_1, N_2=N_2, D_2=D_2)


def DeepNeuralOperatorFactory(trial):
    width = trial.suggest_categorical("width", [32, 64])
    depth = trial.suggest_categorical("depth", [8, 32])
    return partial(DeepNeuralOperator, width=width, depth=depth)


def run_all():
    """Run benchmarks for all operators."""

    # Benchmarks
    benchmarks = [
        SineRegular,
        SineUniform,
    ]

    # Operators
    operators = [
        DeepONetFactory,
        FourierNeuralOperatorFactory,
        BelNetFactory,
        DeepNeuralOperatorFactory,
    ]

    # Run all combinations
    with Pool(n_processes) as p:
        args = [
            (benchmark, operator) for benchmark in benchmarks for operator in operators
        ] * n_trials
        p.starmap(run_single, args)


if __name__ == "__main__":
    run_all()
