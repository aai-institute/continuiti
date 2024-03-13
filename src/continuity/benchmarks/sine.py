"""
`continuity.benchmarks.sine`

Sine benchmark.
"""
import torch
from continuity.data.function import FunctionOperatorDataset, FunctionSet, Function
from continuity.benchmarks import Benchmark
from continuity.discrete import RegularGridSampler


def create_sine_dataset(
    n_observations: int, n_sensors: int, n_evaluations: int
) -> FunctionOperatorDataset:
    r"""Creates a sine dataset.

    A sine dataset is described by the set of trigonometric functions $f(x)_k = \sin(kx)$, with $x\in[-1, 1]$ and
    $k\in[\pi, 2\pi]$. The input space is mapped to the output space with the identity operator (input and output set
    are the same). Both the parameter space, the domain, and the co-domain are sampled on a regular grid.

    Args:
        n_observations: number of observations.
        n_sensors: number of sensors.
        n_evaluations: number of evaluations.

    Returns:
        Dataset of parameterized sine functions mapped by the identity operator.
    """
    sine_set = FunctionSet(lambda k: Function(lambda x: torch.sin(k * x)))

    x_sampler = y_sampler = RegularGridSampler([-1.0], [1.0])
    parameter_sampler = RegularGridSampler([torch.pi], [2 * torch.pi])

    sine_dataset = FunctionOperatorDataset(
        input_function_set=sine_set,
        x_sampler=x_sampler,
        n_sensors=n_sensors,
        output_function_set=sine_set,
        y_sampler=y_sampler,
        n_evaluations=n_evaluations,
        parameter_sampler=parameter_sampler,
        n_observations=n_observations,
    )

    return sine_dataset


def create_sine_benchmark(
    n_observations: int,
    n_sensors: int,
    n_evaluations: int,
    n_test_observations: int,
) -> Benchmark:
    r"""Creates a sine benchmark.

    A sine benchmark is sampled on a regular grid, both in space and parameters. The input function set is mapped onto
    the output function set with the identity operator (input and output function sets are the same). The function sets
    are created with the create_sine_dataset function.

    Args:
        n_observations: number of observations in the train dataset.
        n_sensors: number of sensors.
        n_evaluations: number of evaluation coordinates.
        n_test_observations: number of observations in the test dataset.

    Returns:
        Sine benchmark.
    """

    benchmark = Benchmark(
        train_dataset=create_sine_dataset(
            n_observations=n_observations,
            n_sensors=n_sensors,
            n_evaluations=n_evaluations,
        ),
        test_dataset=create_sine_dataset(
            n_observations=n_test_observations,
            n_sensors=n_sensors,
            n_evaluations=n_evaluations,
        ),
    )

    return benchmark


sine_benchmark = create_sine_benchmark(
    n_observations=90, n_sensors=32, n_evaluations=32, n_test_observations=10
)
