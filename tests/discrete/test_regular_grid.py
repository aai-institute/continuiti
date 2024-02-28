import pytest
import torch
from typing import List

from continuity.discrete import RegularGridSampler


@pytest.fixture(scope="module")
def regular_grid_sampler() -> RegularGridSampler:
    return RegularGridSampler(x_min=torch.zeros((7,)), x_max=torch.ones((7,)))


@pytest.fixture(scope="module")
def regular_grid_random_sampler() -> RegularGridSampler:
    x_min = -2.0 * torch.rand((5,))
    x_max = 2.0 * torch.rand((5,))
    return RegularGridSampler(x_min=x_min, x_max=x_max)


@pytest.fixture(scope="module")
def regular_grid_sampler_under() -> RegularGridSampler:
    return RegularGridSampler(
        x_min=torch.zeros((7,)), x_max=torch.ones((7,)), prefer_more_samples=False
    )


@pytest.fixture(scope="module")
def regular_grid_random_sampler_under() -> RegularGridSampler:
    x_min = -2.0 * torch.rand((5,))
    x_max = 2.0 * torch.rand((5,))
    return RegularGridSampler(x_min=x_min, x_max=x_max, prefer_more_samples=False)


@pytest.fixture(scope="module")
def sampler_list_over(
    regular_grid_sampler, regular_grid_random_sampler
) -> List[RegularGridSampler]:
    return [regular_grid_sampler, regular_grid_random_sampler]


@pytest.fixture(scope="module")
def sampler_list_under(
    regular_grid_sampler_under, regular_grid_random_sampler_under
) -> List[RegularGridSampler]:
    return [regular_grid_sampler_under, regular_grid_random_sampler_under]


@pytest.fixture(scope="module")
def sampler_list(sampler_list_over, sampler_list_under) -> List[RegularGridSampler]:
    return sampler_list_over + sampler_list_under


def test_can_initialize(sampler_list):
    for sampler in sampler_list:
        assert isinstance(sampler, RegularGridSampler)


def test_sample_within_bounds(sampler_list):
    n_samples = 2**12
    for sampler in sampler_list:
        samples = sampler(n_samples)
        samples_min, _ = samples.min(dim=0)
        samples_max, _ = samples.max(dim=0)
        assert torch.greater_equal(samples_min, sampler.x_min).all()
        assert torch.less_equal(samples_max, sampler.x_max).all()


def test_perfect_samples(regular_grid_sampler, regular_grid_sampler_under):
    for sampler in [regular_grid_sampler, regular_grid_sampler_under]:
        n_samples = 10**sampler.ndim
        samples = sampler(n_samples)

        assert samples.size(0) == n_samples


def test_samples_under(sampler_list_under):
    for sampler in sampler_list_under:
        n_samples = 10**sampler.ndim + 1
        samples = sampler(n_samples)

        assert samples.size(0) < n_samples


def test_samples_over(sampler_list_over):
    for sampler in sampler_list_over:
        n_samples = 10**sampler.ndim + 1
        samples = sampler(n_samples)

        assert samples.size(0) > n_samples


def test_dist_zero_single():
    """delta x in a single dimension is zero."""
    n_samples = 121
    sampler = RegularGridSampler(torch.zeros(3), torch.tensor([1.0, 1.0, 0.0]))
    samples = sampler(n_samples)

    assert samples.size(0) == n_samples


def test_dist_zero_double():
    """delta x in multiple dimensions is zero."""
    n_samples = 100
    sampler = RegularGridSampler(torch.zeros(3), torch.tensor([0.0, 1.0, 0.0]))
    samples = sampler(n_samples)

    assert samples.size(0) == n_samples


def test_dist_zero_all():
    """samples are drawn from a single point"""
    n_samples = 100
    sampler = RegularGridSampler(torch.zeros(3), torch.zeros(3))
    samples = sampler(n_samples)

    assert samples.size(0) == n_samples


def test_dist_neg():
    n_samples = 100
    sampler = RegularGridSampler(torch.zeros(3), torch.tensor([0.0, -1.0, 0.0]))
    samples = sampler(n_samples)

    assert samples.size(0) == n_samples
