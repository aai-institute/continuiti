import pytest
import torch
from continuity.discrete.function_set import FunctionSet


@pytest.fixture(scope="module")
def sin_set():
    return FunctionSet(
        lambda a, xi: a[:, 0, None] * torch.sin(a[:, 1, None] * xi + a[:, 2, None])
    )


def test_can_initialize(sin_set):
    assert isinstance(sin_set, FunctionSet)


def test_eval_correct(sin_set):
    n_observations = 10
    x = torch.linspace(-1, 1, 300).repeat(n_observations, 1)

    param = torch.outer(
        torch.ones(
            n_observations,
        ),
        torch.tensor([-2.0, torch.pi, 1.0]),
    )

    assert torch.allclose(
        sin_set(param, x),
        param[:, 0, None] * torch.sin(param[:, 1, None] * x + param[:, 2, None]),
    )
