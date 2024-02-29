import pytest
import torch
from continuity.data.function import FunctionSet
from continuity.data.function import ParameterizedFunction


@pytest.fixture(scope="module")
def lin_p_func():
    return ParameterizedFunction(lambda a, x: a[0] + a[1] * x, n_parameters=2)


@pytest.fixture(scope="module")
def lin_set(lin_p_func):
    return FunctionSet(lin_p_func)


def test_can_initialize(lin_set):
    assert isinstance(lin_set, FunctionSet)


def test_call_correct(lin_p_func, lin_set):
    a = torch.rand(1, 2, 1)
    x = torch.rand(1, 100, 1)

    lf_set = lin_set(a)
    lf = lin_p_func(a)

    for ls, l in zip(lf_set, lf):
        assert torch.allclose(ls(x), l(x))


def test_add_correct(lin_p_func, lin_set):
    union_set = lin_set + lin_set
    union_fun = lin_p_func + lin_p_func

    a = torch.rand(5, union_fun.n_parameters, 1)
    lf_set = union_set(a)
    lf = union_fun(a)

    x = torch.rand(5, 10, 1)
    for ls, l in zip(lf_set, lf):
        assert torch.allclose(ls(x), l(x))


def test_sub_correct(lin_p_func, lin_set):
    union_set = lin_set - lin_set
    union_fun = lin_p_func - lin_p_func

    a = torch.rand(1, union_fun.n_parameters, 1)

    lf_set = union_set(a)
    lf = union_fun(a)

    x = torch.rand(1, 100, 1)
    for ls, l in zip(lf_set, lf):
        assert torch.allclose(ls(x), l(x))
