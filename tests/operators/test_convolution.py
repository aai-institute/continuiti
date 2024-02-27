import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from continuity.data.sine import Sine
from continuity.operators.integralkernel import NaiveIntegralKernel


def test_convolution():
    # Parameters
    num_sensors = 16
    num_evals = num_sensors

    # Data set
    dataset = Sine(num_sensors, size=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    x, u, _, _ = next(iter(dataloader))

    # Kernel
    def dirac(x, y):
        dist = ((x - y) ** 2).sum(dim=-1)
        zero = torch.zeros(1)
        return torch.isclose(dist, zero).to(torch.get_default_dtype())

    # Operator
    operator = NaiveIntegralKernel(
        kernel=dirac,
        coordinate_dim=dataset.coordinate_dim,
        num_channels=dataset.num_channels,
    )

    # Create tensors
    y = torch.linspace(-1, 1, num_evals).reshape(1, -1, 1)

    # Apply operator
    v = operator(x.reshape((1, -1, 1)), u.reshape((1, -1, 1)), y.reshape((1, -1, 1)))

    # Plotting
    fig, ax = plt.subplots(1, 1)
    x_plot = x[0].squeeze().detach().numpy()
    ax.plot(x_plot, u[0].squeeze().detach().numpy(), "x-")
    ax.plot(x_plot, v[0].squeeze().detach().numpy(), "--")
    fig.savefig(f"test_convolution.png")

    # For num_sensors == num_evals, we get v = u / num_sensors.
    if num_sensors == num_evals:
        v_expected = u / num_sensors
        assert (v == v_expected).all(), f"{v} != {v_expected}"


if __name__ == "__main__":
    test_convolution()
