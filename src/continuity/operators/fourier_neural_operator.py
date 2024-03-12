""" Fourier Neural Operator.

TODO:
    - Test what happens if num_modes is not the same for every dimension
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
        num_modes: Optional[Tuple[int]] = None,
    ) -> None:
        super().__init__()

        # TODO: u.dim == v.dim so far -> generalize
        self.shapes = shapes

        assert (
            self.shapes.y.dim == self.shapes.x.dim
        ), f"The dimensions of x and y need to be the same. Given y.dim={self.shapes.y.dim} x.dim={self.shapes.x.dim}"

        points_per_dimension = shapes.x.num ** (1 / shapes.x.dim)
        assert (
            points_per_dimension.is_integer()
        ), "If shapes.x.num ** (1 / shapes.x.dim) is not an integer, the sensor points were not chosen to be on a grid."
        points_per_dimension = int(points_per_dimension)

        self.num_modes = (
            [points_per_dimension] * self.shapes.x.dim
            if num_modes is None
            else num_modes
        )
        self.num_modes = list(self.num_modes)

        # The last dimension is smaller because we use the `torch.fft.rfftn` method.
        # This is due to the negative frequency modes being redundant.
        self.num_modes[-1] = self.num_modes[-1] // 2 + 1

        assert all(mode <= points_per_dimension for mode in self.num_modes[:-1]), (
            "The given number of Fourier modes exceeds the number of sensor points given by dataset.shapes."
            f" Given {self.num_modes[:-1]} Max={points_per_dimension}"
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
        """Forward pass. Performs a kernel integral operation in Fourier space.

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

        # compute n-dimensional real-valued fourier transform
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

        # perform kernel operation in Fourier space
        out_fft = torch.einsum(contraction_equation, self.kernel, u_fft)

        # we should use num_evaluations and not num_sensors for the inverse FFT
        num_evaluations_per_dim = num_evaluations ** (1.0 / num_fft_dimensions)
        assert (
            num_evaluations_per_dim.is_integer()
        ), "Input array 'y' needs to be sampled on a grid with equal number of points per dimension."
        num_evaluations_per_dim = int(num_evaluations_per_dim)

        # target shape for output
        target_shape = (num_evaluations_per_dim,) * (num_fft_dimensions - 1)
        target_shape += (num_evaluations_per_dim // 2 + 1,)

        # add or remove frequencies such that the fft dimensions of out_fft match target_shape
        out_fft = self.add_or_remove_frequencies(
            out_fft, target_shape=target_shape, dim=fft_dimensions
        )

        # transform Fourier modes from 'ascending order' to 'normal order'
        out_fft = self.get_standard_order(out_fft, dim=fft_dimensions)

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
        self, fft_values: torch.Tensor, dim: Tuple[int], target_shape: List[int]
    ) -> torch.Tensor:
        """Remove large positive or negative frequencies.

        This method assumes that the input tensor is in 'ascending order'.
        See `get_ascending_order` for more details.

        Example:
            fft_values.shape -> (10, 32, 17, 2)
            out = remove_large_frequencies(fft_values, dim=(1, 2), target_shape=(20, 11))
            out.shape -> (10, 20, 11, 2)

        Args:
            fft_values: Tensor with fft values.
            dim: List of fft dimensions.
            target_shape:  Determins output shape of fft-dimensions. Should
                be same length as `dim`.

        Returns:
            Tensor with reduced number of fft dimensions.
        """

        assert len(dim) == len(target_shape)

        slices = [slice(None)]  # batch dimension, equivalent to '[:]'

        # loop over fft dimensions except last dimension
        for idx, dimension in enumerate(dim[:-1]):
            num_modes_input = fft_values.shape[dimension]

            # if input is already smaller, don't do anything
            if num_modes_input <= target_shape[idx]:
                slices += [slice(None)]
                continue

            center = num_modes_input // 2
            upper = center + target_shape[idx] // 2 + target_shape[idx] % 2
            # upper = center + int(math.ceil(target_shape[idx] / 2))
            lower = center - target_shape[idx] // 2
            slices += [
                slice(
                    lower,
                    upper,
                )
            ]

        # one sided slicing for last dimension
        slices += [slice(target_shape[-1])]
        slices += [slice(None)]  # u-dim dimension
        u_fourier_sliced = fft_values[slices]

        return u_fourier_sliced

    def get_ascending_order(
        self, fft_values: torch.Tensor, dim: Tuple[int]
    ) -> torch.Tensor:
        """Reorder fft values from 'standard order' to 'ascending order'.

        E.g.:
            Standard order (0, -1, -2, 2, 1) -> Ascending order (-2, -1, 0, 1, 2)

        Note: Last dimension is skipped because when using the real-valued fft
        `torch.fft.rfftn` the last dimension only contains positive frequency
        modes and, therefore, has to be treated separately.

        Args:
            fft_values: tensor with fft values
            dim: List of fft dimensions
        Returns:
            Reordered tensor with ascending order.
        """

        # Last dimension of torch.fft.rfft does not need to be reordered.
        if len(dim) == 1:
            return fft_values

        fft_values_reordered = torch.fft.fftshift(fft_values, dim=dim[:-1])
        return fft_values_reordered

    def get_standard_order(
        self, fft_values: torch.Tensor, dim: Tuple[int]
    ) -> torch.Tensor:
        """Reorder fft values from 'ascending order' to 'standard order'.

        E.g.:
            Ascending order (-2, -1, 0, 1, 2) -> Standard order (0, -1, -2, 2, 1)

        Note: Last dimension is skipped because when using the real-valued fft
        `torch.fft.rfftn` the last dimension only contains positive frequency
        modes and, therefore, has to be treated separately.

        Args:
            fft_values: tensor with fft values
            dim: fft dimensions
        Returns:
            Reordered tensor with standard order.
        """

        if len(dim) == 1:
            return fft_values

        fft_values_reordered = torch.fft.ifftshift(fft_values, dim=dim[:-1])
        return fft_values_reordered

    def zero_padding(
        self, fft_values: torch.Tensor, target_shape: Tuple[int], dim: Tuple[int]
    ) -> torch.Tensor:
        """Add zeros for large positive and negative frequencies.

        This method assumes that the input tensor is in 'ascending order'.
        See `get_ascending_order` for more details.

        Example:
            fft_values.shape -> (10, 32, 17, 2)
            out = zero_padding(fft_values, dim=(1, 2), target_shape=(30, 16))
            out.shape -> (10, 30, 16, 2)

        Args:
            fft_values: tensor with fft values.
            dim: List of fft dimensions.
            target_shape: Determins output shape of fft-dimensions. Should
                be same length as `dim`.

        Returns:
            Tensor with increased number of fft dimensions.
        """

        assert len(target_shape) == len(dim)

        n_dim = fft_values.dim()

        # by default don't add zeros, (left-padding, right-padding)
        padding = [(0, 0)] * n_dim

        # loop over all fft dimensions except last dimension
        for idx, dim_i in enumerate(dim[:-1]):
            # evaluate number of zeros to add
            num_points = fft_values.shape[dim_i]
            zeros_to_add = target_shape[idx] - num_points
            padding_i = (zeros_to_add // 2, zeros_to_add // 2 + zeros_to_add % 2)

            if zeros_to_add < 1:
                continue

            if num_points % 2 == 0:
                # pad more zeros to the right
                padding[dim_i] = padding_i
            else:
                # pad more zeros to the left
                padding[dim_i] = padding_i[::-1]

        # treat last dimension separately
        zeros_to_add = target_shape[-1] - fft_values.shape[dim[-1]]
        if zeros_to_add > 0:
            padding[dim[-1]] = (0, zeros_to_add)

        # `pad`` parameter of `nn.functional.pad` starts with the last dimension -> Reverse order
        padding = padding[::-1]
        padding_flattened = [item for sublist in padding for item in sublist]

        # zero pad input
        fourier_values_zero_padded = torch.nn.functional.pad(
            fft_values, padding_flattened, mode="constant", value=0
        )

        return fourier_values_zero_padded

    def add_or_remove_frequencies(
        self, fft_values: torch.Tensor, target_shape: Tuple[int], dim: Tuple[int]
    ) -> torch.Tensor:
        """Add or remove frequencies depending on target_shape.

        If the dimensions in `target_shape` are larger than the fft dimensions,
            zeros are added as large positive and negative frequencies.
        If the dimensions in `target_shape` are smaller than the fft dimensions,
            large positive and negative frequencies are removed.

        Args:
            fft_values: Tensor with fft values.
            target_shape: Target shape of fft dimensions.
                len(target_shape) == len(dim)
            dim: fft dimensions.
        Returns:
            Tensor with shape:
                (..., target_shape[0], target_shape[1], ..., target_shape[-1], ...)

        """

        # remove large positive and negative frequencies if num_sensors_per_dimension > self.num_modes
        fft_values = self.remove_large_frequencies(
            fft_values, target_shape=target_shape, dim=dim
        )

        # add zero-frequencies if num_sensors_per_dimension < self.num_modes
        fft_values = self.zero_padding(fft_values, target_shape=target_shape, dim=dim)

        return fft_values
