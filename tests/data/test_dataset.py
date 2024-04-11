import torch
from torch.utils.data import DataLoader
from continuiti.data.selfsupervised import SelfSupervisedOperatorDataset


def test_dataset():
    # Sensors
    num_sensors = 4
    f = lambda x: x**2
    num_channels = 1
    coordinate_dim = 1

    x = torch.Tensor(range(num_sensors)).reshape(-1, 1)
    u = f(x)

    # Dataset
    dataset = SelfSupervisedOperatorDataset(x.unsqueeze(0), u.unsqueeze(0))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    x_target, u_target = x, u

    # Test
    for x, u, y, v in dataloader:
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
