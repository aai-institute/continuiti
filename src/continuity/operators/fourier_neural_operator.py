""" Fourier neural operator.

TODO:
    - Assumption: Grid is quadratic -> Should we change this?
    - Generalize kernel to (u-dim, v-dim)
    - x and y are ignored! Should we add options to make kernel dependent?
"""

from continuity.operators import Operator
import torch
from continuity.data import DatasetShapes
import torch.nn as nn
import math
from torch.fft import rfft, irfft, rfftn, irfftn


# TODO: this is not allowed to have 'b', 'd' or 's' in it. How to make this safer?
TENSOR_INDICES = "acefghijklmnopqrtuvxyz"


class FourierLayer1d(Operator):
    """Fourier layer for x.dim = 1.

    Note: This will be removed in the future. See FourierLayer() for general implementation.

    """

    def __init__(
        self,
        shapes: DatasetShapes,
        num_frequencies: int | None = None,
    ) -> None:
        super().__init__()

        assert (
            shapes.x.dim == 1
        ), f"This is the implementation of a 1d Fourier operator. However given dimensionality is x.dim = {shapes.x.dim}"

        self.num_frequencies = (
            shapes.x.num // 2 + 1 if num_frequencies is None else num_frequencies
        )

        assert (
            self.num_frequencies <= shapes.x.num // 2 + 1
        ), "num_frequencies is too large. The fft of a real valued function has only (shapes.x.num // 2 + 1) unique frequencies."

        assert shapes.u.dim == shapes.v.dim

        shape = (self.num_frequencies, shapes.u.dim, shapes.u.dim)
        weigths_real = torch.Tensor(*shape)
        weigths_img = torch.Tensor(*shape)

        # initialize
        nn.init.kaiming_uniform_(weigths_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(weigths_img, a=math.sqrt(5))

        self.weights_complex = torch.complex(weigths_real, weigths_img)
        self.kernel = torch.nn.Parameter(self.weights_complex)

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Sensor positions of shape (batch_size, num_sensors, x_dim)
            u: Input function values of shape (batch_size, num_sensors, u_dim)
            y: Evaluation coordinates of shape (batch_size, num_sensors, y_dim)

        Returns:
            Evaluations of the mapped function with shape (batch_size, num_sensors, v_dim)
        """

        del x, y  # not used

        u_fourier = rfft(u, axis=1)  # real-valued fft -> skip negative frequencies
        out_fourier = torch.einsum(
            "nds,bns->bnd", self.kernel, u_fourier[:, : self.num_frequencies, :]
        )
        out = irfft(out_fourier, axis=1, n=u.shape[1])

        return out


class FourierLayer(Operator):
    """Fourier layer."""

    def __init__(
        self,
        shapes: DatasetShapes,
        num_frequencies: int | None = None,
    ) -> None:
        super().__init__()

        # TODO: u.dim == v.dim so far -> generalize

        self.shapes = shapes

        points_per_dimension = shapes.x.num ** (1 / shapes.x.dim)
        assert (
            points_per_dimension.is_integer()
        ), "If shapes.x.num ** (1 / shapes.x.dim) is not an integer, the sensor points were not chosen to be on a grid."

        # Evaluate the number of frequencies if not specified. By using the rfft method we only need half
        # of the function evaluations because of redundancies for the negative frequencies. This assumes
        # real-valued input functions.
        self.num_frequencies = (
            int(points_per_dimension) // 2 + 1
            if num_frequencies is None
            else num_frequencies
        )
        assert self.num_frequencies <= int(points_per_dimension) // 2 + 1, (
            "num_frequencies is too large. The fft of a real valued function has only (shapes.x.num // 2 + 1) unique values."
            f" Given {self.num_frequencies} Max={int(points_per_dimension) // 2}"
        )

        assert shapes.u.dim == shapes.v.dim  # TODO

        # create and initialize weights and kernel
        weights_shape = (self.num_frequencies,) * self.shapes.x.dim + (
            shapes.u.dim,
            shapes.u.dim,
        )
        weigths_real = torch.Tensor(*weights_shape)
        weigths_img = torch.Tensor(*weights_shape)

        nn.init.kaiming_uniform_(weigths_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(weigths_img, a=math.sqrt(5))

        self.weights_complex = torch.complex(weigths_real, weigths_img)
        self.kernel = torch.nn.Parameter(self.weights_complex)

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Sensor positions of shape (batch_size, num_sensors, x_dim)
            u: Input function values of shape (batch_size, num_sensors, u_dim)
            y: Evaluation coordinates of shape (batch_size, num_evaluations, y_dim)

        Returns:
            Evaluations of the mapped function with shape (batch_size, num_evaluations, v_dim)
        """

        # shapes which can change for different forward passes
        batch_size = y.shape[0]
        num_evaluations = y.shape[1]

        # fourier transform input function
        num_fourier_dimensions = self.shapes.x.dim
        u = self._reshape(u)
        assert u.dim() == num_fourier_dimensions + 2

        # compute n-dimensional fourier transform
        u_fourier = rfftn(
            u, dim=list(range(1, num_fourier_dimensions + 1)), norm="forward"
        )

        assert (
            len(TENSOR_INDICES) > num_fourier_dimensions
        ), f"Too many dimensions. The current limit for the number of dimensions is {len(TENSOR_INDICES)}."

        # contraction equation for torch.einsum method
        # d: v-dim, s: u-dim, b: batch-dim
        frequency_indices = "".join(TENSOR_INDICES[: int(num_fourier_dimensions)])
        contraction_equation = "{}ds,b{}s->b{}d".format(
            frequency_indices, frequency_indices, frequency_indices
        )

        # for the frequency dimensions: cut away high frequencies
        slices = [slice(batch_size)]
        slices += [slice(self.num_frequencies)] * num_fourier_dimensions
        slices += [slice(u.shape[-1])]
        u_fourier_sliced = u_fourier[slices]

        # perform kernel operation in frequency space
        out_fourier = torch.einsum(contraction_equation, self.kernel, u_fourier_sliced)

        # We should use num_evaluations and not num_sensors for the inverse FFT
        num_evaluations_per_dim = num_evaluations ** (1.0 / num_fourier_dimensions)
        assert (
            num_evaluations_per_dim.is_integer()
        ), "Input array 'y' needs to be sampled on a grid with equal number of points per dimension."
        num_evaluations_per_dim = int(num_evaluations_per_dim)

        # transform back into real-space
        out = irfftn(
            out_fourier,
            dim=list(range(1, num_fourier_dimensions + 1)),
            s=(num_evaluations_per_dim,) * num_fourier_dimensions,
            norm="forward",
        )  # TODO: u.shape will change
        out = out.reshape(batch_size, -1, self.shapes.u.dim)

        return out

    def _reshape(self, u: torch.Tensor) -> torch.Tensor:
        """Reshaping input function to make it work with torch.fft.fftn.

        For the FFT we need $u$ to be evaluated on a grid.
        Args:
            u: input function with shape (batch-size, u.num, u.dim).

        Return:
            u: with input function (batch-size, (u.num)**(1/u.dim), (u.num)**(1/u.dim), ..., u.dim)
        """

        # shapes that can change for different forward passes
        batch_size, u_num = u.shape[0], u.shape[1]

        points_per_dim = (u_num) ** (1.0 / self.shapes.x.dim)
        assert (
            points_per_dim.is_integer()
        ), "Input function 'u' can't be reshaped. 'u' needs to be evaluated on a grid."

        shape = [batch_size]
        shape += [int(points_per_dim) for _ in range(self.shapes.x.dim)]
        shape += [self.shapes.u.dim]
        return u.reshape(shape)
