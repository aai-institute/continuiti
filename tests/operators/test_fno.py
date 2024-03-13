import torch
import pytest

from continuity.operators.losses import MSELoss
from continuity.trainer import Trainer
from continuity.data import OperatorDataset
from continuity.operators.fourier_neural_operator import FourierLayer1d, FourierLayer

torch.manual_seed(0)


# @pytest.fixture(scope="module")
def get_dataset() -> OperatorDataset:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Input function
    u = lambda x: torch.sin(x)

    # Target function
    v = lambda x: torch.sin(x)

    # Size parameters
    num_sensors = 32
    num_evaluations = 100

    # Domain parameters
    half_linspace = lambda N: 2 * torch.pi * torch.arange(N) / N
    x = half_linspace(num_sensors).to(device)
    y = half_linspace(num_evaluations).to(device)

    # This dataset contains only a single sample (first dimension of all tensors)
    n_observations = 1
    u_dim = x_dim = y_dim = v_dim = 1
    dataset = OperatorDataset(
        x=x.reshape(n_observations, num_sensors, x_dim),
        u=u(x).reshape(n_observations, num_sensors, u_dim),
        y=y.reshape(n_observations, num_evaluations, y_dim),
        v=v(y).reshape(n_observations, num_evaluations, v_dim),
    )
    return dataset


@pytest.mark.slow
def test_fourier1d():
    dataset = get_dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    operator = FourierLayer1d(dataset.shapes)
    Trainer(operator, device=device).fit(dataset, tol=1e-12, epochs=10_000)

    x, u, y, v = dataset[:1]
    assert MSELoss()(operator, x, u, y, v) < 1e-12


@pytest.mark.slow
def test_fno():
    dataset = get_dataset()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    operator = FourierLayer(dataset.shapes)
    Trainer(operator, device=device).fit(dataset, tol=1e-12, epochs=10_000)

    x, u, y, v = dataset[:1]
    assert MSELoss()(operator, x, u, y, v) < 1e-12


def test_zero_padding():
    dataset = get_dataset()
    operator = FourierLayer(dataset.shapes)

    #### test behaviour for odd number of frequencies
    # input tensor in 'standard order'
    x = torch.tensor(
        [
            [0.0000, 1.0000, 2.0000],
            [0.1000, 1.1000, 2.1000],
            [0.2000, 1.2000, 2.2000],
            [-0.2000, 0.8000, 1.8000],
            [-0.1000, 0.9000, 1.9000],
        ]
    )

    output = torch.tensor(
        [
            [0.0000, 1.0000, 2.0000, 0.0000],
            [0.1000, 1.1000, 2.1000, 0.0000],
            [0.2000, 1.2000, 2.2000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [-0.2000, 0.8000, 1.8000, 0.0000],
            [-0.1000, 0.9000, 1.9000, 0.0000],
        ]
    )

    # test if zero padding works well together with `fftshift` method
    # if we include zeros at the wrong position, the 'standard order' will be destroyed
    x_shifted = torch.fft.fftshift(x, dim=(0,))
    x_zero_padded = operator.zero_padding(
        x_shifted, target_shape=output.shape, dim=(0, 1)
    )
    x_zero_padded = torch.fft.ifftshift(x_zero_padded, dim=(0,))

    assert torch.all(output == x_zero_padded).item()

    #### test behaviour for even number of frequencies
    # input tensor in 'standard order'
    x = torch.tensor(
        [
            [0.0000, 1.0000, -2.0000],
            [0.1000, 1.1000, -1.9000],
            [-0.2000, 0.8000, -2.2000],
            [-0.1000, 0.9000, -2.1000],
        ]
    )

    output = torch.tensor(
        [
            [0.0000, 1.0000, -2.0000, 0.0000],
            [0.1000, 1.1000, -1.9000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [-0.2000, 0.8000, -2.2000, 0.0000],
            [-0.1000, 0.9000, -2.1000, 0.0000],
        ]
    )

    # test if zero padding works well together with `fftshift` method
    # if we include zeros at the wrong position, the 'standard order' will be destroyed
    x_shifted = torch.fft.fftshift(x, dim=(0,))
    x_zero_padded = operator.zero_padding(
        x_shifted, target_shape=output.shape, dim=(0, 1)
    )
    x_zero_padded = torch.fft.ifftshift(x_zero_padded, dim=(0,))

    assert torch.all(output == x_zero_padded).item()


def test_remove_large_frequencies():
    dataset = get_dataset()
    operator = FourierLayer(dataset.shapes)

    #### test behaviour for odd number of frequencies
    # input tensor in 'standard order'
    # frequencies to be removed are marked with zeros
    x = torch.tensor(
        [
            [0.0000, 1.0000, 2.0000, 0.0000],
            [0.1000, 1.1000, 2.1000, 0.0000],
            [0.2000, 1.2000, 2.2000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [-0.2000, 0.8000, 1.8000, 0.0000],
            [-0.1000, 0.9000, 1.9000, 0.0000],
        ]
    )

    output = torch.tensor(
        [
            [0.0000, 1.0000, 2.0000],
            [0.1000, 1.1000, 2.1000],
            [0.2000, 1.2000, 2.2000],
            [-0.2000, 0.8000, 1.8000],
            [-0.1000, 0.9000, 1.9000],
        ]
    )

    # test if removing frequencies works well together with `fftshift` method
    x_shifted = torch.fft.fftshift(x, dim=(0,))
    x_zero_padded = operator.remove_large_frequencies(
        x_shifted, target_shape=output.shape, dim=(0, 1)
    )
    x_zero_padded = torch.fft.ifftshift(x_zero_padded, dim=(0,))

    assert torch.all(output == x_zero_padded).item()

    #### test behaviour for even number of frequencies
    # input tensor in 'standard order'
    # frequencies to be removed are marked with zeros
    x = torch.tensor(
        [
            [0.0000, 1.0000, -2.0000, 0.0000],
            [0.1000, 1.1000, -1.9000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000],
            [-0.2000, 0.8000, -2.2000, 0.0000],
            [-0.1000, 0.9000, -2.1000, 0.0000],
        ]
    )

    output = torch.tensor(
        [
            [0.0000, 1.0000, -2.0000],
            [0.1000, 1.1000, -1.9000],
            [-0.2000, 0.8000, -2.2000],
            [-0.1000, 0.9000, -2.1000],
        ]
    )

    # test if zero padding works well together with `fftshift` method
    x_shifted = torch.fft.fftshift(x, dim=(0,))
    x_zero_padded = operator.remove_large_frequencies(
        x_shifted, target_shape=output.shape, dim=(0, 1)
    )
    x_zero_padded = torch.fft.ifftshift(x_zero_padded, dim=(0,))

    assert torch.all(output == x_zero_padded).item()


if __name__ == "__main__":
    test_zero_padding()
