"""
`continuity.benchmarks.sine`

Sine benchmarks.
"""
import torch
from continuity.data.function import FunctionOperatorDataset, FunctionSet, Function
from continuity.benchmarks import Benchmark
from continuity.discrete import RegularGridSampler, UniformBoxSampler
from continuity.operators.losses import MSELoss


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


SineRegular = SineBenchmark(
    n_sensors=32,
    n_evaluations=32,
    n_train=1024,
    n_test=32,
    uniform=False,
)

SineUniform = SineBenchmark(
    n_sensors=32,
    n_evaluations=32,
    n_train=4096,
    n_test=128,
    uniform=True,
)
