import pytest
import torch
from typing import List

from continuiti.discrete import UniformBoxSampler


@pytest.fixture(scope="module")
def unit_box_sampler() -> UniformBoxSampler:
    return UniformBoxSampler(x_min=torch.zeros((7,)), x_max=torch.ones((7,)))


@pytest.fixture(scope="module")
def random_box_sampler() -> UniformBoxSampler:
    x_min = -2.0 * torch.rand((5,))
    x_max = 2.0 * torch.rand((5,))
    return UniformBoxSampler(x_min=x_min, x_max=x_max)


@pytest.fixture(scope="module")
def sampler_list(unit_box_sampler, random_box_sampler) -> List[UniformBoxSampler]:
    return [unit_box_sampler, random_box_sampler]


def test_can_initialize(unit_box_sampler):
    assert isinstance(unit_box_sampler, UniformBoxSampler)


def test_sample_within_bounds(sampler_list):
    n_samples = 2**12
    for sampler in sampler_list:
        samples = sampler(n_samples)
        samples_min, _ = samples.flatten(1, -1).min(dim=1)
        samples_max, _ = samples.flatten(1, -1).max(dim=1)
        assert torch.greater_equal(samples_min, sampler.x_min).all()
        assert torch.less_equal(samples_max, sampler.x_max).all()


def test_uniform_distribution(sampler_list):
    n_samples = 2**20
    for sampler in sampler_list:
        sample = sampler(n_samples)

        mean = 0.5 * (sampler.x_min + sampler.x_max)
        assert torch.allclose(sample.mean(dim=1), mean, atol=1e-1)
        assert torch.allclose(
            sample.median(dim=1)[0], mean, atol=1e-1
        )  # skew is zero -> mean === median

        var = sampler.x_delta**2 / 12
        assert torch.allclose(sample.var(dim=1), var, atol=1e-1)


def test_uniform_dist_zero():
    n_samples = 100
    sampler = UniformBoxSampler([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])  # initiate with lists
    samples = sampler(n_samples)
    assert samples.flatten(1, -1).size(1) == n_samples


def test_uniform_dist_neg():
    n_samples = 100
    sampler = UniformBoxSampler(
        torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0])
    )
    samples = sampler(n_samples)
    assert samples.flatten(1, -1).size(1) == n_samples
