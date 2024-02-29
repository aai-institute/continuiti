import pytest
import torch
from typing import List
from operator import add, sub, mul, truediv
from continuity.data.function import ParameterizedFunction


@pytest.fixture(scope="module")
def f_x_squared() -> ParameterizedFunction:
    return ParameterizedFunction(lambda a, x: a[0] * (x - a[1]) ** 2, n_parameters=2)


@pytest.fixture(scope="module")
def f_x() -> ParameterizedFunction:
    return ParameterizedFunction(lambda a, x: a[0] * (x - a[1]), n_parameters=2)


@pytest.fixture(scope="module")
def f_const() -> ParameterizedFunction:
    return ParameterizedFunction(lambda a, x: x * 0.0 + a[0], n_parameters=1)


@pytest.fixture(scope="module")
def f_all(f_x, f_x_squared, f_const) -> List[ParameterizedFunction]:
    return [f_x, f_x_squared, f_const]


def test_can_initialize(f_all):
    for f in f_all:
        assert isinstance(f, ParameterizedFunction)


def test_call_format_correct_x(f_x):
    a = torch.linspace(-1, 1, 3).unsqueeze(1)
    f_x_funcs = f_x(a)
    assert isinstance(f_x_funcs, List)
    assert len(f_x_funcs) == 3


def test_call_format_correct_xs(f_x_squared):
    a = torch.rand(5, 2, 1)
    f_x_squared_funcs = f_x_squared(a)
    assert isinstance(f_x_squared_funcs, List)
    assert len(f_x_squared_funcs) == 5


def test_call_correct_x(f_x):
    a = torch.linspace(-1, 1, 3).unsqueeze(0)
    x = torch.linspace(-5, 5, 100)

    f_x_funcs = f_x(a)

    for ai, fx in zip(a, f_x_funcs):
        assert torch.allclose(fx(x), ai[0] * x)


def test_call_correct_xs(f_x_squared):
    a = torch.rand(5, 2, 1)
    x = torch.linspace(-5, 5, 100)

    f_x_funcs = f_x_squared(a)

    for ai, fx in zip(a, f_x_funcs):
        assert torch.allclose(fx(x), ai[0] * (x - ai[1]) ** 2)


def test_update_param_correct(f_x, f_x_squared, f_const):
    for op in [add, sub, mul, truediv]:
        f_new = op(op(f_x, f_x_squared), f_const)
        assert f_new.n_parameters == 5


def test_add_correct(f_x, f_x_squared, f_const):
    x = torch.linspace(-5, 5, 100)

    f_tot = f_x + f_x_squared + f_const
    f_tot_corr = ParameterizedFunction(
        lambda a, x: a[0] * (x - a[1]) + a[2] * (x - a[3]) ** 2 + a[4], n_parameters=5
    )

    a = torch.rand(10, 5, 1) - 0.5
    f_tot_funcs = f_tot(a)
    f_tot_corr_funcs = f_tot_corr(a)

    for f, fc in zip(f_tot_funcs, f_tot_corr_funcs):
        assert torch.allclose(f(x), fc(x))


def test_sub_correct(f_x, f_x_squared, f_const):
    x = torch.linspace(-5, 5, 100)

    f_tot = f_x - f_x_squared - f_const
    f_tot_corr = ParameterizedFunction(
        lambda a, x: a[0] * (x - a[1]) - a[2] * (x - a[3]) ** 2 - a[4], n_parameters=5
    )

    a = torch.rand(10, 5, 1) - 0.5
    f_tot_funcs = f_tot(a)
    f_tot_corr_funcs = f_tot_corr(a)

    for f, fc in zip(f_tot_funcs, f_tot_corr_funcs):
        assert torch.allclose(f(x), fc(x))


def test_mul_correct(f_x, f_x_squared, f_const):
    x = torch.linspace(-5, 5, 100)

    f_tot = f_x * f_x_squared * f_const
    f_tot_corr = ParameterizedFunction(
        lambda a, x: a[0] * a[2] * a[4] * (x - a[1]) * (x - a[3]) ** 2, n_parameters=5
    )

    a = torch.rand(10, 5, 1) - 0.5
    f_tot_funcs = f_tot(a)
    f_tot_corr_funcs = f_tot_corr(a)

    for f, fc in zip(f_tot_funcs, f_tot_corr_funcs):
        assert torch.allclose(f(x), fc(x))


def test_div_correct(f_x, f_x_squared, f_const):
    x = torch.linspace(
        1, 5, 100
    )  # [1, 5] to prevent singularity with random parameters

    f_tot = (f_x / f_x_squared) / f_const
    f_tot_corr = ParameterizedFunction(
        lambda a, x: (a[0] * (x - a[1])) / (a[2] * (x - a[3]) ** 2) / a[4],
        n_parameters=5,
    )

    a = torch.rand(10, 5, 1)

    f_tot_funcs = f_tot(a)
    f_tot_corr_funcs = f_tot_corr(a)

    for f, fc in zip(f_tot_funcs, f_tot_corr_funcs):
        assert torch.allclose(f(x), fc(x))
