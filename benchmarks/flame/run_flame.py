from functools import partial
from continuiti.benchmarks.run import BenchmarkRunner, RunConfig
from continuiti.benchmarks import Flame
from continuiti.operators import ConvolutionalNeuralNetwork


def run():
    config = RunConfig(
        partial(Flame, upsample=True),
        partial(ConvolutionalNeuralNetwork, kernel_size=15),
        lr=1e-4,
        batch_size=16,
        max_epochs=100,
    )
    BenchmarkRunner.run(config)


if __name__ == "__main__":
    run()
