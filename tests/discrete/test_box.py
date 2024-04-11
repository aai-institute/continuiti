import pytest
import torch

from continuiti.discrete.box_sampler import BoxSampler


@pytest.fixture(scope="module")
def generic_box_sampler():
    class GenericBoxSampler(BoxSampler):
        def __call__(self, n_samples: int) -> torch.Tensor:
            return torch.zeros((n_samples, self.ndim))

    return GenericBoxSampler


@pytest.fixture(scope="module")
def unit_box_sampler(generic_box_sampler):
    return generic_box_sampler(
        torch.zeros(
            5,
        ),
        torch.ones(
            5,
        ),
    )


@pytest.fixture(scope="module")
def random_box_sampler(generic_box_sampler):
    return generic_box_sampler(
        -2.0
        * torch.rand(
            5,
        ),
        2
        * torch.rand(
            5,
        ),
    )


@pytest.fixture(scope="module")
def sampler_list(unit_box_sampler, random_box_sampler):
    return [unit_box_sampler, random_box_sampler]


def test_can_initialize(sampler_list, generic_box_sampler):
    for sampler in sampler_list:
        assert isinstance(sampler, generic_box_sampler)


def test_delta_correct(sampler_list):
    for sampler in sampler_list:
        assert torch.allclose(sampler.x_max - sampler.x_min, sampler.x_delta)


def test_dim_correct(sampler_list):
    for sampler in sampler_list:
        assert sampler.ndim == 5
