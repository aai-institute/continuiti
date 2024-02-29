import pytest
import torch

from continuity.data.function import FunctionSet, ParameterizedFunction
from continuity.data.function_dataset import FunctionOperatorDataset
from continuity.discrete import UniformBoxSampler


@pytest.fixture(scope="module")
def some_input_set():
    return FunctionSet(
        ParameterizedFunction(lambda a, x: a[0] + a[1] * x, n_parameters=2)
    )


@pytest.fixture(scope="module")
def some_output_set():
    return FunctionSet(
        ParameterizedFunction(lambda a, x: a[0] * x**2, n_parameters=1)
    )


@pytest.fixture(scope="module")
def some_function_dataset(some_input_set, some_output_set):
    return FunctionOperatorDataset(
        input_function_set=some_input_set,
        solution_function_set=some_output_set,
        x_sampler=UniformBoxSampler(torch.tensor([1.0]), torch.tensor([2.0])),
        y_sampler=UniformBoxSampler(torch.tensor([-2.0]), torch.tensor([-1.0])),
        p_sampler=UniformBoxSampler(
            torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0])
        ),
        n_sensors=2,
        n_evaluations=3,
        n_observations=5,
    )


def test_can_initialize(some_function_dataset):
    assert isinstance(some_function_dataset, FunctionOperatorDataset)


def test_shape_correct(some_function_dataset):
    assert some_function_dataset.shapes.x.num == 2
    assert some_function_dataset.shapes.x.dim == 1
    assert some_function_dataset.shapes.y.num == 3
    assert some_function_dataset.shapes.y.dim == 1
    assert some_function_dataset.shapes.u.num == 2
    assert some_function_dataset.shapes.u.dim == 1
    assert some_function_dataset.shapes.v.num == 3
    assert some_function_dataset.shapes.v.dim == 1
    assert some_function_dataset.shapes.num_observations == 5


def test_space_samples_inside_bounds(some_function_dataset):
    assert torch.less_equal(
        some_function_dataset.x, 2 * torch.ones(some_function_dataset.x.shape)
    ).all()
    assert torch.greater_equal(
        some_function_dataset.x, 1 * torch.ones(some_function_dataset.x.shape)
    ).all()
    assert torch.less_equal(
        some_function_dataset.y, -torch.ones(some_function_dataset.y.shape)
    ).all()
    assert torch.greater_equal(
        some_function_dataset.y, -2 * torch.ones(some_function_dataset.y.shape)
    ).all()
