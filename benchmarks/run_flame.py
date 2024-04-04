from functools import partial
from continuity.benchmarks.run import BenchmarkRunner, RunConfig
from continuity.benchmarks import Flame
from continuity.operators import (
    ConvolutionalNeuralNetwork,
    FourierNeuralOperator,
    DeepNeuralOperator,
)


def run():
    kwargs = {
        "verbose": True,
        "batch_size": 512,
    }

    config = RunConfig(
        partial(Flame, upsample=True),
        partial(ConvolutionalNeuralNetwork, kernel_size=15),
        lr=1e-4,
        **kwargs,
    )
    BenchmarkRunner.run(config)

    config = RunConfig(
        partial(Flame, upsample=True),
        partial(FourierNeuralOperator, width=4, depth=8),
        lr=1e-3,
        **kwargs,
    )
    BenchmarkRunner.run(config)

    config = RunConfig(
        partial(Flame, upsample=False),
        partial(DeepNeuralOperator, width=16 * 16, depth=32),
        lr=1e-4,
        **kwargs,
    )
    BenchmarkRunner.run(config)


if __name__ == "__main__":
    run()
