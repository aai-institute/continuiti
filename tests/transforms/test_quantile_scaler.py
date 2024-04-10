import pytest
import torch

from continuiti.transforms import QuantileScaler


@pytest.fixture(scope="module")
def random_multiscale_tensor():
    t = torch.rand(97, 37, 7)
    t *= 5
    t = 10**t
    return t


@pytest.fixture(scope="module")
def quantile_scaler(random_multiscale_tensor):
    return QuantileScaler(
        src=random_multiscale_tensor, target_mean=0.0, target_std=1.0, eps=1e-2
    )


class TestQuantileScaler:
    def test_can_initialize(self, quantile_scaler):
        isinstance(quantile_scaler, QuantileScaler)

    def test_forward_shape(self, quantile_scaler, random_multiscale_tensor):
        out = quantile_scaler(random_multiscale_tensor)
        assert out.shape == random_multiscale_tensor.shape

    def test_forward(self, quantile_scaler, random_multiscale_tensor):
        out = quantile_scaler(random_multiscale_tensor)

        dist = torch.distributions.normal.Normal(
            torch.zeros(
                1,
            ),
            torch.ones(
                1,
            ),
        )
        limit = dist.icdf(torch.linspace(1e-2, 1 - 1e-2, 2))

        assert torch.all(torch.greater_equal(out, limit[0]))
        assert torch.all(torch.less_equal(out, limit[1]))

    def test_forward_ood(self, quantile_scaler):
        """out of distribution"""
        exp = (torch.rand(97, 37, 7) - 0.5) * 2 * 10
        t = (torch.rand(97, 37, 7) - 0.5) * 2
        t = t**exp

        _ = quantile_scaler(t)
        assert True

    def test_forward_zero_dim(self, quantile_scaler):
        """out of distribution"""
        t = torch.rand(7)

        out = quantile_scaler.undo(t)
        assert out.shape == t.shape

    def test_forward_many_dim(self, quantile_scaler):
        """out of distribution"""
        t = torch.rand(1, 2, 3, 4, 7)

        out = quantile_scaler.undo(t)
        assert out.shape == t.shape

    def test_undo_shape(self, quantile_scaler, random_multiscale_tensor):
        out = quantile_scaler.undo(random_multiscale_tensor)
        assert out.shape == random_multiscale_tensor.shape

    def test_undo(self, quantile_scaler, random_multiscale_tensor):
        out = quantile_scaler(random_multiscale_tensor)
        undone = quantile_scaler.undo(out)
        assert torch.allclose(undone, random_multiscale_tensor, atol=1e-5)

    def test_undo_ood(self, quantile_scaler, random_multiscale_tensor):
        """out of distribution"""
        dist = torch.distributions.normal.Normal(
            torch.zeros(
                1,
            ),
            torch.ones(
                1,
            ),
        )
        limit = dist.icdf(torch.linspace(1e-2, 1 - 1e-2, 2))
        limit *= 10  # max and min by src dist
        test_tensor = torch.linspace(*limit, 700).reshape(1, 100, 7)
        _ = quantile_scaler.undo(test_tensor)
        assert True

    def test_undo_zero_dim(self, quantile_scaler):
        """out of distribution"""
        t = torch.rand(7)

        out = quantile_scaler.undo(t)
        assert out.shape == t.shape

    def test_undo_many_dim(self, quantile_scaler):
        """out of distribution"""
        t = torch.rand(1, 2, 3, 4, 7)

        out = quantile_scaler.undo(t)
        assert out.shape == t.shape
