import pytest
import torch
from typing import List
from continuiti.data.function.function import Function


@pytest.fixture(scope="module")
def f_x_squared() -> Function:
    return Function(lambda x: x**2)


@pytest.fixture(scope="module")
def f_x() -> Function:
    return Function(lambda x: x)


@pytest.fixture(scope="module")
def f_sin() -> Function:
    return Function(torch.sin)


@pytest.fixture(scope="module")
def f_many() -> Function:
    return Function(lambda a, b, c, x: ((a + b) / c) * x)


@pytest.fixture(scope="module")
def f_all(f_x, f_x_squared, f_sin, f_many) -> List[Function]:
    return [f_x, f_x_squared, f_sin, f_many]


def test_can_initialize(f_all):
    for f in f_all:
        assert isinstance(f, Function)


def test_call_correct(f_x, f_x_squared, f_sin, f_many):
    x = torch.linspace(-5, 5, 100)

    assert torch.allclose(f_x(x), x)
    assert torch.allclose(f_x_squared(x), x**2)
    assert torch.allclose(f_sin(x), torch.sin(x))
    assert torch.allclose(f_many(1.0, 1.0, 1.0, x), 2.0 * x)


def test_add_correct(f_x, f_x_squared):
    x = torch.linspace(-5, 5, 100)

    f_tot = f_x + f_x_squared

    assert torch.allclose(f_tot(x), x + x**2)


def test_mul_correct(f_x):
    x = torch.linspace(-5, 5, 100)

    f_tot = -2.0 * f_x

    assert torch.allclose(f_tot(x), -2 * x)
