from typing import List
from continuity.benchmarks.runner import BenchmarkRunner, RunConfig
from continuity.benchmarks.database import BenchmarkDatabase
from continuity.benchmarks.sine import SineRegular, SineUniform
from continuity.operators import (
    DeepONet,
    BelNet,
    FourierNeuralOperator,
    DeepNeuralOperator,
)


def all_runs():
    runs: List[RunConfig] = []

    # Benchmarks
    benchmarks = {
        "SineRegular": lambda: SineRegular,
        "SineUniform": lambda: SineUniform,
    }

    # Operators
    operators = {
        "DeepONet": lambda s: DeepONet(s),
        "FNO": lambda s: FourierNeuralOperator(s),
        "BelNet": lambda s: BelNet(s),
        "DNO": lambda s: DeepNeuralOperator(s),
    }

    # Seeds
    num_seeds = 1

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
                )
                runs.append(run)

    return runs


if __name__ == "__main__":
    db = BenchmarkDatabase()
    runner = BenchmarkRunner()

    for i, run in enumerate(all_runs()):
        print(f"Running {i+1}/{len(all_runs())}")
        stats = runner.run(run)
        db.add_run(stats)
