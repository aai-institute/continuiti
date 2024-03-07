import pytest
import torch
from continuity.data.function_dataset import FunctionOperatorDataset, FunctionSet
from continuity.discrete import RegularGridSampler


@pytest.fixture(scope="module")
def x2_set() -> FunctionOperatorDataset:
    x2 = FunctionSet(lambda a, xi: a[:, None] * xi**2)
    dx2_x = FunctionSet(lambda a, xi: 2 * a[:, None] * xi)

    domain_sampler = RegularGridSampler(torch.tensor([-1.0]), torch.tensor([1.0]))
    parameter_sampler = RegularGridSampler(torch.tensor([-1.0]), torch.tensor([1.0]))

    return FunctionOperatorDataset(
        input_function_set=x2,
        x_sampler=domain_sampler,
        n_sensors=42,
        solution_function_set=dx2_x,
        y_sampler=domain_sampler,
        n_evaluations=13,
        p_sampler=parameter_sampler,
        n_observations=7,
    )


def test_can_initialize(x2_set):
    assert isinstance(x2_set, FunctionOperatorDataset)


def test_correct_shapes(x2_set):
    shapes = x2_set.shapes

    assert shapes.x.dim == 1
    assert shapes.x.num == 42
    assert shapes.u.dim == 1
    assert shapes.u.num == 42
    assert shapes.y.dim == 1
    assert shapes.y.num == 13
    assert shapes.v.dim == 1
    assert shapes.v.num == 13
    assert shapes.num_observations == 7


def test_generate_in_distribution_observation_repr(x2_set):
    x_s, u_s, y_s, v_s = x2_set.generate_observations(
        n_sensors=42, n_evaluations=13, n_observations=7
    )

    assert torch.allclose(x_s, x2_set.x)
    assert torch.allclose(u_s, x2_set.u)
    assert torch.allclose(y_s, x2_set.y)
    assert torch.allclose(v_s, x2_set.v)


def test_generate_in_distribution_observation(x2_set):
    x_s, u_s, y_s, v_s = x2_set.generate_observations(
        n_sensors=2, n_evaluations=3, n_observations=5
    )

    # shapes
    assert x_s.shape == torch.Size([5, 2, 1])
    assert u_s.shape == torch.Size([5, 2, 1])
    assert y_s.shape == torch.Size([5, 3, 1])
    assert v_s.shape == torch.Size([5, 3, 1])

    # in distribution
    assert torch.greater_equal(x_s, -1).all()
    assert torch.less_equal(x_s, 1).all()
    assert torch.greater_equal(y_s, -1).all()
    assert torch.less_equal(y_s, 1).all()
    assert torch.greater_equal(u_s, -1).all()
    assert torch.less_equal(u_s, 1).all()
    assert torch.greater_equal(v_s, -2).all()
    assert torch.less_equal(v_s, 2).all()
    assert torch.greater_equal(v_s, 1).any()
    assert torch.less_equal(v_s, -1).any()


def test_generate_out_of_distribution_observation(x2_set):
    x_sampler = RegularGridSampler(torch.tensor([10.0]), torch.tensor([11.0]))
    y_sampler = RegularGridSampler(torch.tensor([-10.0]), torch.tensor([-9.0]))
    p_sampler = RegularGridSampler(torch.tensor([42.0]), torch.tensor([42.0]))

    x_s, u_s, y_s, v_s = x2_set.generate_observations(
        n_sensors=2,
        n_evaluations=3,
        n_observations=5,
        x_sampler=x_sampler,
        y_sampler=y_sampler,
        p_sampler=p_sampler,
    )

    assert torch.greater_equal(x_s, 10.0).all()
    assert torch.less_equal(x_s, 11.0).all()
    assert torch.greater_equal(y_s, -10).all()
    assert torch.less_equal(y_s, -9).all()
    assert torch.greater_equal(u_s, 42 * 10**2).all()
    assert torch.less_equal(u_s, 42 * 11**2).all()
    assert torch.greater_equal(v_s, 2 * 42 * -10).all()
    assert torch.less_equal(v_s, 2 * 42 * -9).all()
