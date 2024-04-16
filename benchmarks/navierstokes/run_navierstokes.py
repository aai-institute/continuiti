import torch
from functools import partial
from continuiti.benchmarks.run import BenchmarkRunner, RunConfig
from continuiti.benchmarks import NavierStokes
from continuiti.operators import FourierNeuralOperator

# FFT not available on MPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = RunConfig(
    benchmark_factory=NavierStokes,
    operator_factory=partial(
        FourierNeuralOperator,
        width=32,
        depth=4,
    ),
    lr=1e-3,
    max_epochs=100,
    batch_size=10,
    device=device,
)

if __name__ == "__main__":
    BenchmarkRunner.run(config)
