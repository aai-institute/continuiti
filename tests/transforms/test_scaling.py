import pytest
import torch

from continuity.transforms import ZNormalization, Standardization


def test_z_norm_zero():
    mean = torch.zeros((1, 3))
    std = torch.rand((1, 3))

    tf = ZNormalization(mean=mean, std=std)

    assert torch.allclose(tf(torch.zeros(100, 3)), torch.zeros(100, 3))


def test_z_norm_unchanged():
    mean = torch.zeros((1, 3))
    std = torch.ones((1, 3))
    tf = ZNormalization(mean=mean, std=std)

    t = torch.rand((100, 3))

    assert torch.allclose(tf(t), t)


def test_z_norm_mean():
    mean = torch.ones((1, 3))
    std = torch.ones((1, 3))
    tf = ZNormalization(mean=mean, std=std)

    t = torch.rand((100, 3))
    assert torch.allclose(tf(t), t - 1.0)


def test_z_norm_std():
    factor = 2.0
    mean = torch.zeros((1, 3))
    std = factor * torch.ones((1, 3))
    tf = ZNormalization(mean=mean, std=std)

    t = torch.rand((100, 3))
    assert torch.allclose(tf(t), t / factor)


def test_z_norm_correct():
    t = torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).to(
        torch.get_default_dtype()
    )
    mean = torch.mean(t)
    std = torch.std(
        t, correction=0
    )  # correction as otherwise it defaults to Bessel's correction
    assert mean == 5.0
    assert std == 2.0
    tf = ZNormalization(mean, std)

    assert torch.allclose(tf(t), (t - 5.0) / 2.0)


def test_z_norm_singular():
    mean = torch.zeros((1, 3))
    std = torch.zeros((1, 3))
    with pytest.warns(UserWarning) as record:
        tf = ZNormalization(mean=mean, std=std)
    assert len(record) == 1

    t = torch.rand((100, 3))

    assert not torch.any(torch.isnan(tf(t)))


def test_z_norm_alias():
    assert Standardization == ZNormalization
