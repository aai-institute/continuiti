import pytest
import torch


@pytest.fixture
def random_tensor():
    return torch.rand((2, 3, 5, 7)) - 0.5


def test_transform_forward(plus_one_transform, abs_transform, random_tensor):
    assert torch.allclose(
        plus_one_transform(random_tensor),
        random_tensor + torch.ones(random_tensor.shape),
    )
    assert torch.allclose(abs_transform(random_tensor), torch.abs(random_tensor))


def test_transform_backward(plus_one_transform, random_tensor):
    assert torch.allclose(
        plus_one_transform.backward(random_tensor),
        random_tensor - torch.ones(random_tensor.shape),
    )


def test_transform_backward_not_bijective(abs_transform, random_tensor):
    assert torch.allclose(abs_transform.backward(random_tensor), random_tensor)
