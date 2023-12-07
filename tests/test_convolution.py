import torch
import matplotlib.pyplot as plt
from continuity.data.datasets import Sine
from continuity.model.operators import ContinuousConvolution
from continuity.plotting import plot_observation

# Set random seed
torch.manual_seed(0)


def test_convolution():
    # Parameters
    num_sensors = 16
    num_evals = num_sensors

    # Observation
    dataset = Sine(num_sensors, size=1)
    observation = dataset.get_observation(0)

    # Kernel
    def dirac(x, y):
        dist = ((x - y) ** 2).sum(dim=-1)
        return torch.isclose(dist, torch.zeros(1)).to(torch.float32)

    # Operator
    operator = ContinuousConvolution(
        coordinate_dim=dataset.coordinate_dim,
        num_channels=dataset.num_channels,
        kernel=dirac,
    )

    # Create tensors
    yu = observation.to_tensor()
    x = torch.linspace(-1, 1, num_evals).unsqueeze(-1)

    # Turn into batch
    yu = yu.unsqueeze(0)
    x = x.unsqueeze(0)

    # Apply operator
    v = operator(yu, x)

    # Extract batch
    x = x.squeeze(0)
    v = v.squeeze(0)

    # Plotting
    fig, ax = plt.subplots(1, 1)
    plot_observation(observation, ax=ax)
    plt.plot(x, v, "o")
    fig.savefig(f"test_convolution.png")

    # For num_sensors == num_evals, we get v = u / num_sensors.
    if num_sensors == num_evals:
        v_expected = yu[:, :, -1:] / num_sensors
        assert (v == v_expected).all(), f"{v} != {v_expected}"


if __name__ == "__main__":
    test_convolution()
