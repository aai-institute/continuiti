import pytest
import torch
from continuiti.data.function.function_set import FunctionSet, Function


@pytest.fixture(scope="module")
def sin_set_nested_lmbda():
    return FunctionSet(lambda a: lambda xi: a[0] * torch.sin(a[1] * xi + a[2]))


@pytest.fixture(scope="module")
def sin_set_nested_func():
    return FunctionSet(
        lambda a: Function(lambda xi: a[0] * torch.sin(a[1] * xi + a[2]))
    )


def test_can_initialize(sin_set_nested_lmbda, sin_set_nested_func):
    assert isinstance(sin_set_nested_func, FunctionSet)
    assert isinstance(sin_set_nested_lmbda, FunctionSet)


def test_eval_correct(sin_set_nested_lmbda, sin_set_nested_func):
    x = torch.linspace(-1, 1, 300).unsqueeze(0)
    param = torch.tensor([-2.0, torch.pi, 1.0]).unsqueeze(1).repeat(1, 7)

    for func in [sin_set_nested_func, sin_set_nested_lmbda]:
        functions = func(param)
        f = torch.stack([function(x) for function in functions])
        assert torch.allclose(
            f,
            param[0, :, None, None]
            * torch.sin(param[1, :, None, None] * x + param[2, :, None, None]),
        )


def test_call_type_correct(sin_set_nested_lmbda, sin_set_nested_func):
    n_observations = 10
    torch.linspace(-1, 1, 300).repeat(n_observations, 1)
    param = torch.tensor([-2.0, torch.pi, 1.0]).repeat(n_observations, 1)

    for p_func in [sin_set_nested_func, sin_set_nested_lmbda]:
        functions = p_func(param)
        for func in functions:
            assert isinstance(func, Function)
