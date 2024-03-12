""" Fourier neural operator.

TODO:
    - Assumption: Grid is quadratic -> Should we change this?
    - Generalize kernel to (u-dim, v-dim)
    - x and y are ignored! Should we add options to make kernel dependent?
"""
# %%
from continuity.operators import Operator
import torch
from continuity.data import DatasetShapes
import torch.nn as nn
import math
from torch.fft import rfft, irfft, rfftn, irfftn
from typing import Optional, Tuple, List


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
        num_modes: Optional[Tuple[int]] = None,
    ) -> None:
        super().__init__()

        # TODO: u.dim == v.dim so far -> generalize
        self.shapes = shapes

        assert (
            self.shapes.y.dim == self.shapes.x.dim
        ), f"The dimensions of x and y need to be the same. Given y.dim={self.shapes.y.dim} x.dim={self.shapes.x.dim}"

        self.points_per_dimension = shapes.x.num ** (1 / shapes.x.dim)
        assert (
            self.points_per_dimension.is_integer()
        ), "If shapes.x.num ** (1 / shapes.x.dim) is not an integer, the sensor points were not chosen to be on a grid."
        self.points_per_dimension = int(self.points_per_dimension)

        if num_modes is None:
            self.num_modes = [self.points_per_dimension] * self.shapes.x.dim

        # The last dimension is smaller because we use the `torch.fft.rfftn` method.
        # This is due to the negative frequency modes being redundant.
        self.num_modes[-1] = self.num_modes[-1] // 2 + 1

        assert all(mode <= self.points_per_dimension for mode in self.num_modes[:-1]), (
            "The given number of Fourier modes exceeds the number of sensor points given by dataset.shapes."
            f" Given {self.num_modes[:-1]} Max={self.points_per_dimension}"
        )

        assert shapes.u.dim == shapes.v.dim  # TODO

        weights_shape = self.num_modes + [shapes.u.dim, shapes.v.dim]

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
        num_evaluations = y.shape[1]

        # fft related parameters
        num_fft_dimensions = self.shapes.x.dim
        fft_dimensions = list(range(1, num_fft_dimensions + 1))

        # reshape input to prepare for FFT
        u = self._reshape(u)
        assert u.dim() == num_fft_dimensions + 2

        # compute n-dimensional real-values fourier transform
        u_fft = rfftn(u, dim=fft_dimensions, norm="forward")

        assert (
            len(TENSOR_INDICES) > num_fft_dimensions
        ), f"Too many dimensions. The current limit for the number of dimensions is {len(TENSOR_INDICES)}."

        # contraction equation for torch.einsum method
        # d: v-dim, s: u-dim, b: batch-dim
        frequency_indices = "".join(TENSOR_INDICES[: int(num_fft_dimensions)])
        contraction_equation = "{}ds,b{}s->b{}d".format(
            frequency_indices, frequency_indices, frequency_indices
        )

        # transform Fourier modes from 'normal order' to 'ascending order'
        u_fft = self.get_ascending_order(u_fft, dim=fft_dimensions)

        # add or remove frequencies such that the the fft dimensions of u_fft match self.num_modes
        u_fft = self.add_or_remove_frequencies(
            u_fft, target_shape=self.num_modes, dim=fft_dimensions
        )

        # Perform kernel operation in Fourier space
        out_fft = torch.einsum(contraction_equation, self.kernel, u_fft)

        # We should use num_evaluations and not num_sensors for the inverse FFT
        num_evaluations_per_dim = num_evaluations ** (1.0 / num_fft_dimensions)
        assert (
            num_evaluations_per_dim.is_integer()
        ), "Input array 'y' needs to be sampled on a grid with equal number of points per dimension."
        num_evaluations_per_dim = int(num_evaluations_per_dim)

        # Target shape for output
        target_shape = (num_evaluations_per_dim,) * (num_fft_dimensions - 1)
        target_shape += (num_evaluations_per_dim // 2 + 1,)

        # add or remove frequencies such that the fft dimensions of out_fft match target_shape
        out_fft = self.add_or_remove_frequencies(
            out_fft, target_shape=target_shape, dim=fft_dimensions
        )

        # transform Fourier modes from 'ascending order' to 'normal order'
        out_fft = self.get_normal_order(out_fft, dim=fft_dimensions)

        # transform back into real-space
        out = irfftn(
            out_fft,
            dim=fft_dimensions,
            s=(num_evaluations_per_dim,) * num_fft_dimensions,
            norm="forward",
        )

        # TODO: u.shape will change
        out = out.reshape(batch_size, -1, self.shapes.u.dim)

        return out

    def _reshape(self, u: torch.Tensor) -> torch.Tensor:
        """Reshape input function from flattened respresentation to
        grid representation. The grid representation is required for
        `torch.fft.fftn`.

        Assumes that u was evaluated on a grid with the same number of points
        for each dimension.

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

    def remove_large_frequencies(
        self, u_fourier: torch.Tensor, dim: Tuple[int], num_modes: List[int]
    ) -> torch.Tensor:
        slices = [slice(None)]  # batch dimension

        # add fft dimensions except last dimension
        for idx, dimension in enumerate(dim[:-1]):
            num_modes_input = u_fourier.shape[dimension]

            if num_modes_input <= num_modes[idx]:
                slices += [slice(None)]
            else:
                center = num_modes_input // 2
                upper = center + int(math.ceil(num_modes[idx] / 2))
                lower = center - num_modes[idx] // 2
                slices += [
                    slice(
                        lower,
                        upper,
                    )
                ]

        slices += [slice(num_modes[-1])]  # last fft dimension
        slices += [slice(None)]  # u-dim dimension
        u_fourier_sliced = u_fourier[slices]

        return u_fourier_sliced

    def get_ascending_order(self, fourier_values, dim) -> torch.Tensor:
        # reorder frequencies to have largest negativ frequency on the left side and
        # largest positive frequency on the right side. E.g.: (0, -1, -2, 2, 1) -> (-2, -1, 0, -1, -2)
        # skip last dimension because last dimension is rfft related

        # Last dimension of torch.fft.rfft does not need to be reordered.
        if len(dim) == 1:
            return fourier_values

        fourier_values_reordered = torch.fft.fftshift(fourier_values, dim=dim[:-1])
        return fourier_values_reordered

    def get_normal_order(self, fourier_values, dim):
        if len(dim) == 1:
            return fourier_values

        fourier_values_reordered = torch.fft.ifftshift(fourier_values, dim=dim[:-1])
        return fourier_values_reordered

    def zero_padding(
        self, fft_values: torch.Tensor, target_shape: Tuple[int], dim: Tuple[int]
    ) -> torch.Tensor:
        assert len(target_shape) == len(dim)

        n_dim = fft_values.dim()

        # reorder frequencies to have largest negativ frequency on the left side and
        # largest positive frequency on the right side. E.g.: (0, -1, -2, 2, 1) -> (-2, -1, 0, -1, -2)
        # fourier_values_reordered = torch.fft.fftshift(fourier_values, dim=dims_to_reorder)

        # new implementation
        padding = [(0, 0)] * n_dim

        for idx, dim_i in enumerate(dim[:-1]):
            zeros_to_add = target_shape[idx] - fft_values.shape[dim_i]
            padding_zeros = (zeros_to_add // 2, zeros_to_add // 2 + zeros_to_add % 2)
            N = fft_values.shape[dim_i]

            if zeros_to_add < 1:
                continue

            if N % 2 == 0:
                padding[dim_i] = padding_zeros
            else:
                padding[dim_i] = padding_zeros[
                    ::-1
                ]  # if odd, pad more zeros to the left

        # treat last dimension special
        zeros_to_add = target_shape[-1] - fft_values.shape[dim[-1]]
        if zeros_to_add > 0:
            padding[dim[-1]] = (0, zeros_to_add)

        padding = padding[::-1]
        padding_flattened = [item for sublist in padding for item in sublist]

        fourier_values_zero_padded = torch.nn.functional.pad(
            fft_values, padding_flattened, mode="constant", value=0
        )

        return fourier_values_zero_padded

    def add_or_remove_frequencies(
        self, fft_values: torch.Tensor, target_shape: Tuple[int], dim: Tuple[int]
    ) -> torch.Tensor:
        # remove large positive and negative frequencies if num_sensors_per_dimension > self.num_modes
        fft_values = self.remove_large_frequencies(
            fft_values, num_modes=target_shape, dim=dim
        )

        # add zero-frequencies if num_sensors_per_dimension < self.num_modes
        fft_values = self.zero_padding(fft_values, target_shape=target_shape, dim=dim)

        return fft_values
