import torch
import matplotlib.pyplot as plt
from continuity.data import tensor
from continuity.data.datasets import SelfSupervisedDataSet
from continuity.plotting import plot
from torch.utils.data import DataLoader

# Set random seed
torch.manual_seed(0)


def test_dataset():
    # Sensors
    num_sensors = 4
    f = lambda x: x**2
    num_channels = 1
    coordinate_dim = 1

    x = tensor(range(num_sensors)).reshape(-1, 1)
    u = f(x)

    # Test plotting
    fig, ax = plt.subplots(1, 1)
    plot(x, u, ax=ax)
    fig.savefig(f"test_dataset.png")

    # Dataset
    dataset = SelfSupervisedDataSet(
        x.unsqueeze(0),
        u.unsqueeze(0),
    )
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    x_target, u_target = x, u

    # Test
    for sample in dataloader:
        x, u, y, v = sample

        # Every x, u must equal observation
        assert x.shape[1] == num_sensors
        assert u.shape[1] == num_sensors
        assert x.shape[2] == coordinate_dim
        assert u.shape[2] == num_channels
        assert (x == x_target).all()
        assert (u == u_target).all()

        # Every y, v must equal f(y)
        assert y.shape[1] == coordinate_dim
        assert v.shape[1] == num_channels
        assert (v == f(y)).all()


if __name__ == "__main__":
    test_dataset()
