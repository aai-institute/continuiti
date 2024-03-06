import torch
import pytest

from continuity.operators.losses import MSELoss
from continuity.trainer import Trainer
from continuity.data import OperatorDataset
from continuity.operators.fourier_neural_operator import FourierLayer


@pytest.fixture(scope="module")
def dataset() -> OperatorDataset:
    # Input function
    u = lambda x: torch.sin(x)

    # Target function
    v = lambda x: torch.sin(x)

    # Size parameters
    num_sensors = 32
    num_evaluations = 100

    # Domain parameters
    x = torch.linspace(0, 2 * torch.pi, num_sensors)
    y = torch.linspace(0, 2 * torch.pi, num_evaluations)

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
def test_fno(dataset):
    operator = FourierLayer(dataset.shapes)

    Trainer(operator).fit(dataset, tol=1e-12)

    x, u, y, v = dataset[:1]
    assert MSELoss()(operator, x, u, y, v) < 1e-12
