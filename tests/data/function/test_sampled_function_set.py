import pytest
import torch
from continuity.data.function import FunctionSet, SampledFunctionSet
from continuity.data.function import SampledFunctionSet, ParameterizedFunction


@pytest.fixture(scope="module")
def lin_p_func():
    return ParameterizedFunction(lambda a, x: a[0] + a[1] * x, n_parameters=2)


@pytest.fixture(scope="module")
def lin_set(lin_p_func):
    return FunctionSet(function=lin_p_func)


@pytest.fixture(scope="module")
def sam_lin_set(lin_set):
    samples = torch.outer(
        torch.arange(10),
        torch.ones(
            3,
        ),
    )
    return SampledFunctionSet(function_set=lin_set, samples=samples)


def test_can_initialize(sam_lin_set):
    assert isinstance(sam_lin_set, SampledFunctionSet)


def test_call_correct(sam_lin_set):
    x = torch.linspace(-1, 1, 3).repeat(10, 1, 1)

    out = sam_lin_set(x)

    assert out.shape == x.shape

    for i in range(10):
        corr_out = i * (x[0] + 1)
        assert torch.allclose(out[i, :], corr_out)
