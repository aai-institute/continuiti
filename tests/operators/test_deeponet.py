import torch
import matplotlib.pyplot as plt

from continuity.plotting import plot, plot_evaluation
from torch.utils.data import DataLoader
from continuity.operators import DeepONet
from continuity.data import OperatorDataset, Sine


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

    model = DeepONet(dset.shape)

    x, u, y, v = dset[:batch_size]

    v_pred = model(x, u, y)

    assert v_pred.shape == v.shape

    y_other = torch.rand((batch_size, n_evals * 5, y_dim))
    v_other = torch.rand((batch_size, n_evals * 5, v_dim))

    v_other_pred = model(x, u, y_other)

    assert v_other_pred.shape == v_other.shape


def test_deeponet():
    # Parameters
    num_sensors = 16

    # Data set
    dataset = Sine(num_sensors, size=1)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Operator
    operator = DeepONet(
        dataset.shape,
        branch_width=32,
        branch_depth=1,
        trunk_width=32,
        trunk_depth=1,
        basis_functions=4,
    )

    # Train self-supervised
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-2)
    operator.compile(optimizer)
    operator.fit(data_loader, epochs=1000)

    # Plotting
    fig, ax = plt.subplots(1, 1)
    x, u, _, _ = dataset[0]
    plot(x, u, ax=ax)
    plot_evaluation(operator, x, u, ax=ax)
    fig.savefig(f"test_deeponet.png")

    # Check solution
    x = x.unsqueeze(0)
    u = u.unsqueeze(0)
    assert operator.loss(x, u, x, u) < 1e-3
