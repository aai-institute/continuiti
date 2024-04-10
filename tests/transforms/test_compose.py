import pytest
import torch

from continuiti.transforms import Compose


@pytest.fixture
def random_negative_tensor():
    return -torch.rand((2, 3, 5, 7))


@pytest.fixture(scope="module")
def plus_one_times_two(plus_one_transform, times_two_transform) -> Compose:
    return Compose([plus_one_transform, times_two_transform])


def test_compose_forward(random_negative_tensor, plus_one_times_two):
    assert torch.allclose(
        plus_one_times_two(random_negative_tensor), (random_negative_tensor + 1.0) * 2.0
    )


def test_compose_backward(random_negative_tensor, plus_one_times_two):
    transformed_tensor = plus_one_times_two(random_negative_tensor)

    assert torch.allclose(
        plus_one_times_two.undo(transformed_tensor), random_negative_tensor
    )
