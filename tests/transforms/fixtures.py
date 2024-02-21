import pytest
import torch
import warnings

from continuity.transforms import Transform


@pytest.fixture(scope="module")
def plus_one_transform() -> Transform:
    class PlusOne(Transform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor + torch.ones(tensor.shape)

        def undo(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor - torch.ones(tensor.shape)

    return PlusOne()


@pytest.fixture(scope="module")
def times_two_transform() -> Transform:
    class TimesTwo(Transform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor * 2.0

        def undo(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor / 2.0

    return TimesTwo()


@pytest.fixture(scope="module")
def abs_transform() -> Transform:
    class Abs(Transform):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return torch.abs(tensor)

        def undo(self, tensor: torch.Tensor) -> torch.Tensor:
            warnings.warn(
                f"The {self.__class__.__name__} transformation is not bijective. "
                f"Returns the identity instead!",
                stacklevel=2,
            )
            return tensor

    return Abs()
