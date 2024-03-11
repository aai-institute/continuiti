import torch
import pytest

from continuity.operators.losses import MSELoss
from continuity.trainer import Trainer
from continuity.data import OperatorDataset
from continuity.operators.fourier_neural_operator import FourierLayer1d, FourierLayer

torch.manual_seed(0)


# @pytest.fixture(scope="module")
def dataset() -> OperatorDataset:
    # Input function
    u = lambda x: torch.sin(x)

    # Target function
    v = lambda x: torch.sin(x)

    # Size parameters
    num_sensors = 32
    num_evaluations = 100

    # Domain parameters
    half_linspace = lambda N: 2 * torch.pi * torch.arange(N) / N
    x = half_linspace(num_sensors)
    y = half_linspace(num_evaluations)

    # This dataset contains only a single sample (first dimension of all tensors)
    n_observations = 1
    u_dim = x_dim = y_dim = v_dim = 1
    dataset = OperatorDataset(
        x=x.reshape(n_observations, num_sensors, x_dim),
        u=u(x).reshape(n_observations, num_sensors, u_dim),
        y=y.reshape(n_observations, num_evaluations, y_dim),
        v=v(y).reshape(n_observations, num_evaluations, v_dim),
    )
    return dataset


@pytest.mark.slow
def test_fourier1d(dataset):
    operator = FourierLayer1d(dataset.shapes)
    Trainer(operator).fit(dataset, tol=1e-12, epochs=10_000)

    x, u, y, v = dataset[:1]
    assert MSELoss()(operator, x, u, y, v) < 1e-12


@pytest.mark.slow
def test_fno(dataset):
    operator = FourierLayer(dataset.shapes)
    Trainer(operator).fit(dataset, tol=1e-12, epochs=10_000)

    x, u, y, v = dataset[:1]
    assert MSELoss()(operator, x, u, y, v) < 1e-12


if __name__ == "__main__":
    dataset = dataset()
    test_fourier1d(dataset)
