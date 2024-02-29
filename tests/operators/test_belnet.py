import torch
import matplotlib.pyplot as plt
import pytest

from continuity.plotting import plot, plot_evaluation
from continuity.operators import BelNet
from continuity.data import OperatorDataset
from continuity.data.sine import Sine
from continuity.trainer import Trainer
from continuity.operators.losses import MSELoss


def test_belnet_shape():
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

    model = BelNet(dset.shapes)

    x, u, y, v = dset[:batch_size]

    v_pred = model(x, u, y)

    assert v_pred.shape == v.shape

    y_other = torch.rand((batch_size, n_evals, y_dim))
    v_other = torch.rand((batch_size, n_evals, v_dim))

    v_other_pred = model(x, u, y_other)

    assert v_other_pred.shape == v_other.shape


@pytest.mark.slow
def test_belnet():
    # Parameters
    num_sensors = 16

    # Data set
    dataset = Sine(n_sensors=num_sensors, n_observations=1)

    # Operator
    operator = BelNet(
        dataset.shapes,
    )

    # Train self-supervised
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3)
    trainer = Trainer(operator, optimizer)
    trainer.fit(dataset, epochs=1000, batch_size=1, shuffle=True)

    # Plotting
    fig, ax = plt.subplots(1, 1)
    x, u, _, _ = dataset[0]
    plot(x, u, ax=ax)
    plot_evaluation(operator, x, u, ax=ax)
    fig.savefig(f"test_belnet.png")

    # Check solution
    x = x.unsqueeze(0)
    u = u.unsqueeze(0)
    assert MSELoss()(operator, x, u, x, u) < 1e-2


if __name__ == "__main__":
    test_belnet_shape()
    test_belnet()
