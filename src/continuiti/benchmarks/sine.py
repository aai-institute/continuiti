"""
`continuiti.benchmarks.sine`

Sine benchmarks.
"""
import torch
from continuiti.data.function import FunctionOperatorDataset, FunctionSet, Function
from continuiti.benchmarks import Benchmark
from continuiti.discrete import RegularGridSampler, UniformBoxSampler
from continuiti.operators.losses import MSELoss


class SineBenchmark(Benchmark):
    r"""Sine benchmark.

    The `SineBenchmark` contains a dataset of trigonometric functions

    $$
        f_k(x) = \sin(kx), \quad x \in [-1, 1], \quad k \in [\pi, 2\pi],
    $$

    with the following properties:

    - Input and output function spaces are the same.
    - The input space is mapped to the output space with the identity operator.
    - Both the the domain and the co-domain are sampled on a regular grid, if
      `uniform` is `False`, or uniformly, otherwise.
    - The parameter $k$ is sampled uniformly from $[\pi, 2\pi]$.

    Args:
        n_sensors: number of sensors.
        n_evaluations: number of evaluations.
        n_train: number of observations in the train dataset.
        n_test: number of observations in the test dataset.
        uniform: whether to sample the domain and co-domain random uniformly.

    """

    def __init__(
        self,
        n_sensors: int = 32,
        n_evaluations: int = 32,
        n_train: int = 1024,
        n_test: int = 32,
        uniform: bool = False,
    ):
        sine_set = FunctionSet(lambda k: Function(lambda x: torch.sin(k * x)))

        if uniform:
            x_sampler = y_sampler = UniformBoxSampler([-1.0], [1.0])
        else:
            x_sampler = y_sampler = RegularGridSampler([-1.0], [1.0])

        parameter_sampler = UniformBoxSampler([torch.pi], [2 * torch.pi])

        def get_dataset(n_obs: int):
            return FunctionOperatorDataset(
                input_function_set=sine_set,
                x_sampler=x_sampler,
                n_sensors=n_sensors,
                output_function_set=sine_set,
                y_sampler=y_sampler,
                n_evaluations=n_evaluations,
                parameter_sampler=parameter_sampler,
                n_observations=n_obs,
            )

        train_dataset = get_dataset(n_train)
        test_dataset = get_dataset(n_test)

        super().__init__(train_dataset, test_dataset, [MSELoss()])


class SineRegular(SineBenchmark):
    r"""Sine benchmark with the domain and co-domain sampled on a regular grid.

    The `SineRegular` benchmark is a `SineBenchmark` with the following
    properties:

    - `n_sensors` is 32.
    - `n_evaluations` is 32.
    - `n_train` is 1024.
    - `n_test` is 1024.
    - `uniform` is `False`.

    ![Visualizations of a few samples.](/continuiti/benchmarks/img/SineRegular.svg)

    """

    def __init__(self):
        super().__init__(
            n_sensors=32,
            n_evaluations=32,
            n_train=1024,
            n_test=1024,
            uniform=False,
        )


class SineUniform(SineBenchmark):
    r"""Sine benchmark with the domain and co-domain sampled random uniformly.

    The `SineRegular` benchmark is a `SineBenchmark` with the following
    properties:

    - `n_sensors` is 32.
    - `n_evaluations` is 32.
    - `n_train` is 4096.
    - `n_test` is 4096.
    - `uniform` is `True`.

    ![Visualizations of a few samples.](/continuiti/benchmarks/img/SineUniform.svg)

    """

    def __init__(self):
        super().__init__(
            n_sensors=32,
            n_evaluations=32,
            n_train=4096,
            n_test=4096,
            uniform=True,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n = 3

    for benchmark in [SineRegular(), SineUniform()]:
        fig, axs = plt.subplots(n, 2, figsize=(10, 4 * n))
        indices = torch.randint(0, len(benchmark.train_dataset), (n,))
        for i in range(n):
            idx = indices[i]
            x, u, y, v = benchmark.train_dataset[idx]
            axs[i][0].scatter(x, u, 10, color="green")
            axs[i][0].set_title(f"Input {idx}")
            axs[i][1].scatter(y, v, 10, color="red")
            axs[i][1].set_title(f"Output {idx}")

        fig.tight_layout()
        fig.savefig(f"docs/benchmarks/{benchmark}.svg", transparent=True)
