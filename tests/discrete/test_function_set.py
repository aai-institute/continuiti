import pytest
import torch
from continuity.discrete.function_set import FunctionSet


@pytest.fixture(scope="module")
def sin_set():
    return FunctionSet(lambda a, xi: torch.cos(a[:, None] * xi))


def test_can_initialize(sin_set):
    assert isinstance(sin_set, FunctionSet)


def test_eval_correct(sin_set):
    x = torch.linspace(-1, 1, 300)
    assert torch.allclose(sin_set(torch.pi, x), torch.sin(torch.pi * x))
