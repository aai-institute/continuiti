"""
`continuity.benchmarks.sine`

Sine benchmark.
"""
import torch
from continuity.data.function import FunctionOperatorDataset, FunctionSet, Function
from continuity.benchmarks import Benchmark
from continuity.discrete import RegularGridSampler
from continuity.operators.losses import MSELoss


class SineBenchmark(Benchmark):
    r"""Sine benchmark.

    The sine benchmark contains a dataset of trigonometric functions

    $$
        f(x)_k = \sin(kx)$, with $x\in[-1, 1]
    $$

    and $k\in[\pi, 2\pi]$.

    The input space is mapped to the output space with the identity operator
    (input and output set are the same). Both the parameter space, the domain,
    and the co-domain are sampled on a regular grid.

    Args:
        n_sensors: number of sensors.
        n_evaluations: number of evaluations.
        n_train: number of observations in the train dataset.
        n_test: number of observations in the test dataset.

    """

    def __init__(
        self,
        n_sensors: int = 32,
        n_evaluations: int = 32,
        n_train: int = 90,
        n_test: int = 10,
    ):
        sine_set = FunctionSet(lambda k: Function(lambda x: torch.sin(k * x)))

        x_sampler = y_sampler = RegularGridSampler([-1.0], [1.0])
        parameter_sampler = RegularGridSampler([torch.pi], [2 * torch.pi])

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
