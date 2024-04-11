import pytest
import torch

from continuiti.transforms import Normalize


@pytest.fixture(scope="module")
def random_normalization_set():
    mean = torch.rand((1, 1, 15, 7))
    std = torch.rand((1, 1, 15, 7))
    tf = Normalize(mean=mean, std=std)
    t = torch.rand((100, 3, 15, 7))
    return tf, t


def test_normalization_zero():
    mean = torch.zeros((1, 3))
    std = torch.rand((1, 3))

    tf = Normalize(mean=mean, std=std)

    assert torch.allclose(tf(torch.zeros(100, 3)), torch.zeros(100, 3))


def test_normalization_unchanged():
    mean = torch.zeros((1, 3))
    std = torch.ones((1, 3))
    tf = Normalize(mean=mean, std=std)

    t = torch.rand((100, 3))

    assert torch.allclose(tf(t), t)


def test_normalization_mean():
    mean = torch.ones((1, 3))
    std = torch.ones((1, 3))
    tf = Normalize(mean=mean, std=std)

    t = torch.rand((100, 3))
    assert torch.allclose(tf(t), t - 1.0)


def test_normalization_std():
    factor = 2.0
    mean = torch.zeros((1, 3))
    std = factor * torch.ones((1, 3))
    tf = Normalize(mean=mean, std=std)

    t = torch.rand((100, 3))
    assert torch.allclose(tf(t), t / factor)


def test_normalization_correct():
    t = torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).to(
        torch.get_default_dtype()
    )
    mean = torch.mean(t)
    std = torch.std(
        t, correction=0
    )  # correction as otherwise it defaults to Bessel's correction
    assert mean == 5.0
    assert std == 2.0
    tf = Normalize(mean, std)

    assert torch.allclose(tf(t), (t - 5.0) / 2.0)


def test_normalization_singular():
    mean = torch.zeros((1, 3))
    std = torch.zeros((1, 3))
    tf = Normalize(mean=mean, std=std)

    t = torch.rand((100, 3))
    tf_t = tf(t)

    assert not torch.any(torch.isnan(tf_t))
    assert not torch.any(torch.isinf(tf_t))


def test_normalization_dimensions(random_normalization_set):
    tf, t = random_normalization_set
    assert tf(t).shape == t.shape


def test_normalization_undo(random_normalization_set):
    tf, t = random_normalization_set

    t_normalized = tf(t)
    t2 = tf.undo(t_normalized)

    # higher tolerance required because of numerical accuracy of these operations
    assert torch.allclose(t, t2, atol=1e-7)
