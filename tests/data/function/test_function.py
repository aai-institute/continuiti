import pytest
import torch
from typing import List
from continuity.data.function import Function


@pytest.fixture(scope="module")
def f_x_squared() -> Function:
    return Function(lambda x: x**2)


@pytest.fixture(scope="module")
def f_x() -> Function:
    return Function(lambda x: x)


@pytest.fixture(scope="module")
def f_const() -> Function:
    return Function(lambda x: x * 0.0 + 1)


@pytest.fixture(scope="module")
def f_all(f_x, f_x_squared, f_const) -> List[Function]:
    return [f_x, f_x_squared, f_const]


def test_can_initialize(f_all):
    for f in f_all:
        assert isinstance(f, Function)


def test_call_correct(f_x, f_x_squared):
    x = torch.linspace(-5, 5, 100)

    assert torch.allclose(f_x(x), x)
    assert torch.allclose(f_x_squared(x), x**2)


def test_add_correct(f_x, f_x_squared, f_const):
    x = torch.linspace(-5, 5, 100)

    f_tot = f_x + f_x_squared + f_const

    assert torch.allclose(f_tot(x), x + x**2 + 1)


def test_sub_correct(f_x, f_x_squared, f_const):
    x = torch.linspace(-5, 5, 100)

    f_tot = f_x - f_x_squared - f_const

    assert torch.allclose(f_tot(x), x - x**2 - 1)


def test_mul_correct(f_x, f_x_squared, f_const):
    x = torch.linspace(-5, 5, 100)

    f_tot = f_x * f_x_squared * f_const

    assert torch.allclose(f_tot(x), x**3)


def test_div_correct(f_x, f_x_squared, f_const):
    x = torch.linspace(-5, 5, 100)

    f_tot = (f_x_squared / f_x) / f_const

    assert torch.allclose(f_tot(x), x)
