from functools import partial
from continuity.benchmarks.run import BenchmarkRunner, RunConfig
from continuity.benchmarks import NavierStokes
from continuity.operators import FourierNeuralOperator

config = RunConfig(
    benchmark_factory=NavierStokes,
    operator_factory=partial(
        FourierNeuralOperator,
        grid_shape=(64, 64, 10),
        width=32,
        depth=4,
    ),
    lr=1e-3,
    max_epochs=100,
    batch_size=10,
)

if __name__ == "__main__":
    BenchmarkRunner.run(config)
