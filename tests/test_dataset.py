import torch
import matplotlib.pyplot as plt
from continuity.data import Sensor, Observation
from continuity.data.dataset import SelfSupervisedDataSet
from continuity.plotting.plotting import *

# Set random seed
torch.manual_seed(0)


def test_dataset():
    # Sensors
    num_sensors = 4
    f = lambda x: x**2
    num_channels = 1
    coordinate_dim = 1

    sensors = []
    for i in range(num_sensors):
        x = np.array([i])
        u = f(x)
        sensor = Sensor(x, u)
        sensors.append(sensor)

    # Observation
    observation = Observation(sensors)
    print(observation)

    # Test plotting
    fig, ax = plt.subplots(1, 1)
    plot_observation(observation, ax=ax)
    fig.savefig(f"test_dataset.png")

    # Dataset
    dataset = SelfSupervisedDataSet(
        [observation],
        batch_size=3,
    )

    # Test
    for i in range(len(dataset)):
        u, x, v = dataset[i]

        # Every u must equal observation
        assert u.shape[1] == num_sensors
        assert u.shape[2] == num_channels + coordinate_dim
        assert (u == observation.to_tensor()).all()

        # Every v must equal f(x)
        assert v.shape[1] == num_channels
        assert x.shape[1] == coordinate_dim
        assert (v == f(x)).all()


if __name__ == "__main__":
    test_dataset()
