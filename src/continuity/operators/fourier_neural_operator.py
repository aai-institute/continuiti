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
from typing import Optional, Tuple


# TODO: this is not allowed to have 'b', 'd' or 's' in it. How to make this safer?
TENSOR_INDICES = "acefghijklmnopqrtuvxyz"


class FourierLayer1d(Operator):
    """Fourier layer for x.dim = 1.

    Note: This will be removed in the future. See FourierLayer() for general implementation.

    """

    def __init__(
        self,
        shapes: DatasetShapes,
        num_frequencies: Optional[int] = None,
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
        weights_real = torch.Tensor(*shape)
        weights_img = torch.Tensor(*shape)

        # initialize
        nn.init.kaiming_uniform_(weights_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(weights_img, a=math.sqrt(5))

        self.weights_complex = torch.complex(weights_real, weights_img)
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

        u_fourier = rfft(u, axis=1, norm="forward")

        out_fourier = torch.einsum(
            "nds,bns->bnd", self.kernel, u_fourier[:, : self.num_frequencies, :]
        )

        out = irfft(out_fourier, axis=1, n=y.shape[1], norm="forward")

        return out


class FourierLayer(Operator):
    """Fourier layer."""

    def __init__(
        self,
        shapes: DatasetShapes,
        num_frequencies: Optional[int] = None,
    ) -> None:
        super().__init__()

        # TODO: u.dim == v.dim so far -> generalize
        self.shapes = shapes

        self.points_per_dimension = shapes.x.num ** (1 / shapes.x.dim)
        assert (
            self.points_per_dimension.is_integer()
        ), "If shapes.x.num ** (1 / shapes.x.dim) is not an integer, the sensor points were not chosen to be on a grid."

        # Evaluate the number of frequencies if not specified. By using the rfft method we only need half
        # of the function evaluations for the last dimension because of redundancies for the negative frequencies. This assumes
        # real-valued input functions.
        self.num_frequencies = (
            int(self.points_per_dimension) // 2 + 1
            if num_frequencies is None
            else num_frequencies
        )

        assert self.num_frequencies <= int(self.points_per_dimension) // 2 + 1, (
            "num_frequencies is too large. The fft of a real valued function has only (shapes.x.num // 2 + 1) unique values."
            f" Given {self.num_frequencies} Max={int(self.points_per_dimension) // 2}"
        )

        assert shapes.u.dim == shapes.v.dim  # TODO

        # create and initialize weights and kernel
        weights_shape = ()
        if self.shapes.x.dim > 1:
            weights_shape += (int(self.points_per_dimension),) * (self.shapes.x.dim - 1)
        weights_shape += (self.num_frequencies, shapes.u.dim, shapes.v.dim)

        weights_real = torch.Tensor(*weights_shape)
        weights_img = torch.Tensor(*weights_shape)

        nn.init.kaiming_uniform_(weights_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(weights_img, a=math.sqrt(5))

        self.weights_complex = torch.complex(weights_real, weights_img)
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
        num_evaluations, num_sensors = y.shape[1], u.shape[1]

        # fourier transform input function
        num_fourier_dimensions = self.shapes.x.dim
        u = self._reshape(u)
        assert u.dim() == num_fourier_dimensions + 2

        frequency_dimensions = list(range(1, num_fourier_dimensions + 1))

        # compute n-dimensional fourier transform
        u_fourier = rfftn(u, dim=frequency_dimensions, norm="forward")

        assert (
            len(TENSOR_INDICES) > num_fourier_dimensions
        ), f"Too many dimensions. The current limit for the number of dimensions is {len(TENSOR_INDICES)}."

        # contraction equation for torch.einsum method
        # d: v-dim, s: u-dim, b: batch-dim
        frequency_indices = "".join(TENSOR_INDICES[: int(num_fourier_dimensions)])
        contraction_equation = "{}ds,b{}s->b{}d".format(
            frequency_indices, frequency_indices, frequency_indices
        )

        if num_fourier_dimensions > 1:
            u_fourier = self.get_ascending_order(u_fourier, dim=frequency_dimensions)

        # u_fourier_sliced = u_fourier

        slices = [slice(batch_size)]
        if num_fourier_dimensions > 1:
            points_per_dimension = u_fourier.shape[1]
            num_frequencies = points_per_dimension // 2 + 1

            if num_frequencies <= self.num_frequencies:
                slices += [slice(None)]
            else:
                upper = points_per_dimension // 2
                upper += int(math.ceil(self.points_per_dimension / 2))
                slices += [
                    slice(
                        points_per_dimension // 2 - self.points_per_dimension // 2,
                        upper,
                    )
                ]
        slices += [slice(self.num_frequencies)]
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

        if num_evaluations_per_dim > out_fourier.shape[1]:
            out_fourier = self.zero_padding(
                out_fourier,
                out_fourier.shape[1],
                num_evaluations_per_dim - out_fourier.shape[1],
                dim=list(range(1, num_fourier_dimensions + 1)),
            )

        if num_fourier_dimensions > 1:
            out_fourier = self.get_normal_order(out_fourier, dim=frequency_dimensions)

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

    def get_ascending_order(self, fourier_values, dim):
        # reorder frequencies to have largest negativ frequency on the left side and
        # largest positive frequency on the right side. E.g.: (0, -1, -2, 2, 1) -> (-2, -1, 0, -1, -2)
        # skip last dimension because last dimension is rfft related
        fourier_values_reordered = torch.fft.fftshift(fourier_values, dim=dim[:-1])
        return fourier_values_reordered

    def get_normal_order(self, fourier_values, dim):
        fourier_values_reordered = torch.fft.ifftshift(fourier_values, dim=dim[:-1])
        return fourier_values_reordered

    def zero_padding(
        self, fourier_values: torch.Tensor, N: int, zeros_to_add: int, dim: Tuple[int]
    ) -> torch.Tensor:
        assert zeros_to_add >= 0

        n_dim = fourier_values.dim()
        dims_to_reorder = dim[:-1]

        # Special case -> just zero pad last dimension
        # if len(dim) == 1 and rfft_mode:

        #    return torch.nn.functional.pad(fourier_values, (0, zeros_to_add//2+1), mode='constant', value=0)

        # reorder frequencies to have largest negativ frequency on the left side and
        # largest positive frequency on the right side. E.g.: (0, -1, -2, 2, 1) -> (-2, -1, 0, -1, -2)
        # fourier_values_reordered = torch.fft.fftshift(fourier_values, dim=dims_to_reorder)

        # if odd number, one side has to be padded with more zeros
        padding_zeros = (zeros_to_add // 2, zeros_to_add // 2 + zeros_to_add % 2)

        padding = []
        for dim_i in range(n_dim)[::-1]:
            if dim_i in dim[:-1]:  # exclude last dimension
                if N % 2 == 0:
                    padding.extend(padding_zeros)
                else:
                    padding.extend(
                        padding_zeros[::-1]
                    )  # if odd, padd more zeros to the left
            else:
                if dim_i == dim[-1]:  # check for last dimension
                    padding.extend((0, zeros_to_add // 2 + 1))
                else:
                    padding.extend((0, 0))

        # print("padding", padding)
        fourier_values_zero_padded = torch.nn.functional.pad(
            fourier_values, padding, mode="constant", value=0
        )

        # convert back to 'normal order'
        # fourier_values_zero_padded = torch.fft.ifftshift(fourier_values_zero_padded, dim=dims_to_reorder)

        # zero pad last dimension
        # if rfft_mode:
        #    fourier_values_zero_padded = torch.nn.functional.pad(fourier_values_zero_padded, (0, zeros_to_add//2+1), mode='constant', value=0)

        return fourier_values_zero_padded
