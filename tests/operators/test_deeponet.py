import torch
import pytest

from continuity.operators import DeepONet
from continuity.data import OperatorDataset
from continuity.data.sine import Sine
from continuity.trainer import Trainer
from continuity.operators.losses import MSELoss


def test_output_shape():
    x_dim = 2
    u_dim = 3
    y_dim = 5
    v_dim = 7
    n_sensors = 11
    n_evals = 13
    batch_size = 17
    set_size = 19

    dset = OperatorDataset(
        x=torch.rand((set_size, n_sensors, x_dim)),
        u=torch.rand((set_size, n_sensors, u_dim)),
        y=torch.rand((set_size, n_evals, y_dim)),
        v=torch.rand((set_size, n_evals, v_dim)),
    )

    model = DeepONet(dset.shapes)

    x, u, y, v = dset[:batch_size]

    v_pred = model(x, u, y)

    assert v_pred.shape == v.shape

    y_other = torch.rand((batch_size, n_evals * 5, y_dim))
    v_other = torch.rand((batch_size, n_evals * 5, v_dim))

    v_other_pred = model(x, u, y_other)

    assert v_other_pred.shape == v_other.shape


@pytest.mark.slow
def test_deeponet():
    # Parameters
    num_sensors = 16

    # Data set
    dataset = Sine(num_sensors, size=1)

    # Operator
    operator = DeepONet(
        dataset.shapes,
        branch_width=32,
        branch_depth=1,
        trunk_width=32,
        trunk_depth=1,
        basis_functions=4,
    )

    # Train
    Trainer(operator).fit(dataset, tol=1e-3, batch_size=1)

    # Check solution
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 1e-3


if __name__ == "__main__":
    test_output_shape()
    test_deeponet()
