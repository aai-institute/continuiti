import torch
from functools import partial
from continuiti.benchmarks.run import BenchmarkRunner, RunConfig
from continuiti.benchmarks import NavierStokes
from continuiti.operators import FourierNeuralOperator

# FFT not available on MPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

runs = []
for depth in [4, 8, 16]:
    config = RunConfig(
        benchmark_factory=NavierStokes,
        operator_factory=partial(
            FourierNeuralOperator,
            width=32,
            depth=depth,
        ),
        max_epochs=500,
        batch_size=10,
        device=device,
    )
    params_dict = {"depth": depth}
    runs.append((config, params_dict))

if __name__ == "__main__":
    for config, params_dict in runs:
        BenchmarkRunner.run(config, params_dict)
