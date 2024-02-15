import pytest
import torch

from continuity.transforms import Transform


@pytest.fixture(scope="module")
def plus_one_transform() -> Transform:
    class PlusOne(Transform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor + torch.ones(tensor.shape)

        def backward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor - torch.ones(tensor.shape)

    return PlusOne()


@pytest.fixture(scope="module")
def times_two_transform() -> Transform:
    class TimesTwo(Transform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor * 2.0

        def backward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor / 2.0

    return TimesTwo()


@pytest.fixture(scope="module")
def abs_transform() -> Transform:
    class Abs(Transform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.abs(tensor)

    return Abs()
