import torch
from torch.utils.data import DataLoader
from torch.nn.functional import conv1d, conv2d
from torch.fft import irfftn, irfft
import matplotlib.pyplot as plt
import pytest
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

from continuity.operators.losses import MSELoss
from continuity.plotting import plot
from continuity.trainer import Trainer
from continuity.data import OperatorDataset
from continuity.operators.fourier_neural_operator import FourierLayer

torch.manual_seed(0)


def get_1d_dataset(device, num_sensors=64):
    # Input function
    u = lambda chi: torch.exp(-(chi**2))

    # target function
    v = lambda chi: 2 * torch.exp(-(chi**2)) * (-chi)

    # Domain parameters
    L = torch.pi * 2
    x = torch.linspace(-L, L, num_sensors).to(device)
    y = torch.linspace(-L, L, num_sensors).to(device)

    # This dataset contains only a single sample (first dimension of all tensors)
    n_observations = 1
    u_dim = x_dim = y_dim = v_dim = 1
    dataset = OperatorDataset(
        x=x.reshape(n_observations, num_sensors, x_dim),
        u=u(x).reshape(n_observations, num_sensors, u_dim),
        y=y.reshape(n_observations, num_sensors, y_dim),
        v=v(y).reshape(n_observations, num_sensors, v_dim),
    )
    return dataset


def get_2d_dataset(device, num_sensors=64):
    x = torch.linspace(-3, 3, num_sensors)
    y = torch.linspace(-3, 3, num_sensors)
    xx, yy = torch.meshgrid(x, y)
    input_tensor = torch.stack([xx.flatten(), yy.flatten()], axis=1)

    loc, scale = 0, 1
    distribution = MultivariateNormal(torch.ones(2) * loc, scale * torch.eye(2))
    u = torch.exp(distribution.log_prob(input_tensor))
    u /= u.max()

    dataset = OperatorDataset(
        input_tensor.reshape(1, -1, 2),
        u.reshape(1, -1, 1),
        input_tensor.reshape(1, -1, 2),
        u.reshape(1, -1, 1),
    )
    return dataset


def test_fourierlayer_1d():
    """Test if FourierLayer output is identical to convolution in real space.

    This could fail if any of the FFT operations are performed along the wrong axis.
    """

    num_sensors = 64

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define 1d dataset
    dataset = get_1d_dataset(device, num_sensors=num_sensors)
    xi, ui, yi, _ = dataset[:]

    # define layer
    fourier_layer = FourierLayer(dataset.shapes)

    # compute fourier trafo in real space and convolution
    # conv1d does actually perform a cross-correlation -> flip kernel
    kernel_normal_space = irfft(fourier_layer.kernel, dim=0, n=ui.shape[1])
    kernel_normal_space = kernel_normal_space.reshape(1, 1, -1).flip(dims=[-1])
    output_convolution = conv1d(
        ui.reshape(1, 1, -1),  # (batch-size, in-channels, conv dimension)
        kernel_normal_space,
        bias=None,
        padding="same",
    )
    output_convolution = output_convolution.detach().numpy().reshape(-1)

    # compute FourierLayer output
    output_layer = fourier_layer(xi, ui, yi).reshape(-1).detach().numpy()

    # To get both operations matching, we have to mirror around the y-axis.
    # TODO: I am not sure why this is the case. It could be due to the relationship between
    # cross correlation and convolution. Maybe flipping the kernel is not enough?
    n_half = int(len(output_layer) / 2)
    output_layer = np.concatenate(
        [output_layer[n_half:], output_layer[:n_half]], axis=0
    )

    # plot results
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        xi.reshape(-1),
        output_layer.reshape(-1),
        label="Output Fourier Layer",
        marker=".",
    )
    ax.plot(
        xi.reshape(-1),
        output_convolution.reshape(-1),
        label="Output Convolution",
        marker=".",
    )
    ax.legend()
    ax.set_xlabel("y")
    ax.set_ylabel("v(y)")
    fig.savefig(f"test_fourierlayer_convolution_1d.png")

    # check difference
    # we have to cut the boundaries because the convolution introduces boundary effects
    n_boundary = 10
    diff = np.abs(
        output_layer[n_boundary : num_sensors - n_boundary]
        - output_convolution[n_boundary : num_sensors - n_boundary]
    )
    diff_relative = diff / np.abs(output_layer[n_boundary : num_sensors - n_boundary])
    assert diff_relative.mean() < 1e-2


def test_fourierlayer_2d():
    """Test if FourierLayer output is identical to convolution in real space.

    This could fail if any of the FFT operations are performed along the wrong axis.
    """

    num_sensors = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define 1d dataset
    dataset = get_2d_dataset(device, num_sensors=num_sensors)
    xi, ui, yi, _ = dataset[:]

    # define layer
    fourier_layer = FourierLayer(dataset.shapes)

    # compute fourier trafo in real space and then 2d convolution
    # conv2d does actually perform a cross-correlation -> flip kernel
    kernel_normal_space = irfftn(
        fourier_layer.kernel, dim=[0, 1], s=(num_sensors, num_sensors)
    )
    kernel_normal_space = kernel_normal_space.reshape(
        1, 1, num_sensors, num_sensors
    ).flip(dims=[-1, -2])
    output_convolution = conv2d(
        ui.reshape(
            1, 1, num_sensors, num_sensors
        ),  # (batch-size, in-channels, conv dimension1, conv dimension2)
        kernel_normal_space,
        bias=None,
        padding="same",
    )
    output_convolution = (
        output_convolution.detach().numpy().reshape(num_sensors, num_sensors)
    )

    # compute FourierLayer output
    output_layer = (
        fourier_layer(xi, ui, yi).reshape(num_sensors, num_sensors).detach().numpy()
    )

    # To get both operations matching, we have to mirror around the y-axis.
    # TODO: I am not sure why this is the case. It could be due to the relationship between
    # cross correlation and convolution. Maybe flipping the kernel is not enough?
    n_half = int(num_sensors / 2)
    output_layer = np.concatenate(
        [output_layer[n_half:, :], output_layer[:n_half, :]], axis=0
    )
    output_layer = np.concatenate(
        [output_layer[:, n_half:], output_layer[:, :n_half]], axis=1
    )

    # plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[12, 4])
    img1 = ax1.imshow(output_layer, label="Output Fourier Layer")
    img2 = ax2.imshow(output_convolution, label="Output Convolution")
    img3 = ax3.imshow(
        np.abs(output_layer - output_convolution), label="Output Convolution"
    )
    ax1.set_title("Output Fourier Layer")
    ax2.set_title("Output Convolution")
    ax3.set_title("Difference")
    fig.colorbar(img1)
    fig.colorbar(img2)
    fig.colorbar(img3)
    fig.savefig(f"test_fourierlayer_convolution_2d.png")

    # check difference
    # we have to cut the boundaries because the convolution introduces boundary effects
    n_boundary = 20
    diff = np.abs(
        output_layer[
            n_boundary : num_sensors - n_boundary, n_boundary : num_sensors - n_boundary
        ]
        - output_convolution[
            n_boundary : num_sensors - n_boundary, n_boundary : num_sensors - n_boundary
        ]
    )
    diff_relative = diff / np.abs(
        output_layer[
            n_boundary : num_sensors - n_boundary, n_boundary : num_sensors - n_boundary
        ]
    )

    assert diff_relative.mean() < 1e-1


@pytest.mark.slow
def test_fourier_layer_performance():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define 1d dataset
    dataset = get_1d_dataset(device, num_sensors=64)
    xi, ui, yi, vi = dataset[:]
    data_loader = DataLoader(dataset)

    # define layer
    fourier_layer = FourierLayer(dataset.shapes)

    # Train self-supervised
    optimizer = torch.optim.Adam(fourier_layer.parameters(), lr=1e-2)
    trainer = Trainer(fourier_layer, device=device, optimizer=optimizer)
    trainer.fit(data_loader, 1000)

    # compute FourierLayer output
    output = fourier_layer(xi, ui, yi).reshape(-1).detach().numpy()

    # plot results
    fig, ax = plt.subplots(1, 1)
    plot(xi[0], vi[0], ax)
    ax.plot(xi.reshape(-1), output.reshape(-1))
    fig.savefig(f"test_fourierlayer_performance.png")

    assert MSELoss()(fourier_layer, xi, ui, yi, vi) < 1e-3


if __name__ == "__main__":
    test_fourierlayer_1d()
    test_fourier_layer_performance()
    test_fourierlayer_2d()


# %%
