import torch
import numpy as np
import matplotlib.pyplot as plt
from continuity.data import Sensor, Observation
from continuity.data.datasets import SelfSupervisedDataSet
from continuity.plotting import plot_observation

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
        x, u, y, v = dataset[i]

        # Every x, u must equal observation
        assert x.shape[1] == num_sensors
        assert u.shape[1] == num_sensors
        assert x.shape[2] == coordinate_dim
        assert u.shape[2] == num_channels
        x_target, u_target = observation.to_tensors()
        assert (x == x_target).all()
        assert (u == u_target).all()

        # Every y, v must equal f(y)
        assert y.shape[1] == coordinate_dim
        assert v.shape[1] == num_channels
        assert (v == f(y)).all()


if __name__ == "__main__":
    test_dataset()
