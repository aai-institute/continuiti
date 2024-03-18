from typing import List
from continuity.benchmarks.runner import RunConfig
from continuity.benchmarks.sine import SineBenchmark
from continuity.operators import DeepONet, BelNet, FourierNeuralOperator


def all_runs():
    runs: List[RunConfig] = []

    # Benchmarks
    benchmarks = {
        "Sine32": lambda: SineBenchmark(32, 32),
        "Sine128": lambda: SineBenchmark(128, 128),
    }

    # Operators
    operators = {
        "DeepONet": lambda s: DeepONet(s),
        "FNO": lambda s: FourierNeuralOperator(s),
        "BelNet": lambda s: BelNet(s),
    }

    # Seeds
    num_seeds = 3

    # Training parameters
    lr = 1e-3
    tol = 1e-4
    max_epochs = 1_000
    device = "cpu"

    # Generate all combinations
    for benchmark_name, benchmark_factory in benchmarks.items():
        for operator_name, operator_factory in operators.items():
            for seed in range(num_seeds):
                run = RunConfig(
                    benchmark_name,
                    benchmark_factory,
                    operator_name,
                    operator_factory,
                    seed,
                    lr,
                    tol,
                    max_epochs,
                    device,
                )
                runs.append(run)

    # Any custom runs here
    # ...

    return runs
