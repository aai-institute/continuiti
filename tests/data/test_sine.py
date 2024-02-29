import pytest
import torch

from continuity.data.sine import Sine
from continuity.discrete import UniformBoxSampler


@pytest.fixture(scope="module")
def sine_set():
    return Sine(
        n_sensors=10,
        n_observations=10,
        x_sampler=UniformBoxSampler(torch.tensor([-1.0]), torch.tensor([1.0])),
        y_sampler=UniformBoxSampler(torch.tensor([-1.0]), torch.tensor([1.0])),
        p_sampler=UniformBoxSampler(
            torch.tensor([0.0, 0.0, 0.0]), torch.tensor([10.0, 10.0, 10.0])
        ),
    )


@pytest.fixture(scope="module")
def const_sine_set():
    return Sine(
        n_sensors=10,
        n_observations=10,
        x_sampler=UniformBoxSampler(torch.tensor([-1.0]), torch.tensor([1.0])),
        y_sampler=UniformBoxSampler(torch.tensor([-1.0]), torch.tensor([1.0])),
        p_sampler=UniformBoxSampler(
            torch.tensor([0.15, 0.42, 0.7]), torch.tensor([0.15, 0.42, 0.7])
        ),
    )


def test_can_initialize(sine_set, const_sine_set):
    assert isinstance(sine_set, Sine)
    assert isinstance(const_sine_set, Sine)


def test_samples_within_bounds(sine_set):
    assert torch.less_equal(sine_set.u, 10 * torch.ones(sine_set.u.shape)).all()
    assert torch.greater_equal(sine_set.u, -10 * torch.ones(sine_set.u.shape)).all()
    assert torch.less_equal(sine_set.v, 10 * torch.ones(sine_set.v.shape)).all()
    assert torch.greater_equal(sine_set.v, -10 * torch.ones(sine_set.v.shape)).all()


def test_samples_correct(const_sine_set):
    correct_out = 0.15 * torch.sin(0.42 * (const_sine_set.x + 0.7))
    assert torch.allclose(correct_out, const_sine_set.u)
