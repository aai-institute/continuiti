import pytest
import torch
from continuity.data.function_dataset.function_set import FunctionSet


@pytest.fixture(scope="module")
def sin_set():
    return FunctionSet(lambda a: lambda xi: a[0] * torch.sin(a[1] * xi + a[2]))


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
    functions = sin_set(param)

    f = torch.stack([function(x) for function in functions])

    assert torch.allclose(
        f,
        param[:, 0, None] * torch.sin(param[:, 1, None] * x + param[:, 2, None]),
    )
