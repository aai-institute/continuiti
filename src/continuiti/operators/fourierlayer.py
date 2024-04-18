"""
`continuiti.operators.fourierlayer`

The Fourier layer of the Fourier Neural Operator (FNO).
"""

import math
import torch
import torch.nn as nn
from torch.fft import rfft, irfft, rfftn, irfftn
from typing import Optional, Tuple, List
from continuiti.operators import Operator
from continuiti.operators.shape import OperatorShapes


class FourierLayer1d(Operator):
    """Fourier layer for `x.dim = 1`.

    Note:
        This implementation is here for reference only.
        See `FourierLayer` for general implementation.

    """

    def __init__(
        self,
        shapes: OperatorShapes,
        num_frequencies: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(shapes, device)

        assert (
            shapes.x.dim == 1
        ), f"This is the implementation of a 1d Fourier operator. However given dimensionality is x.dim = {shapes.x.dim}"

        x_num = math.prod(shapes.x.size)
        self.num_frequencies = (
            x_num // 2 + 1 if num_frequencies is None else num_frequencies
        )

        assert (
            self.num_frequencies <= x_num // 2 + 1
        ), "num_frequencies is too large. The fft of a real valued function has only (shapes.x.num // 2 + 1) unique frequencies."

        assert shapes.u.dim == shapes.v.dim

        shape = (self.num_frequencies, shapes.u.dim, shapes.u.dim)
        weights_real = torch.empty(shape, device=device)
        weights_img = torch.empty(shape, device=device)

        # initialize
        nn.init.kaiming_uniform_(weights_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(weights_img, a=math.sqrt(5))

        self.weights_complex = torch.complex(weights_real, weights_img)
        self.kernel = torch.nn.Parameter(
            torch.view_as_real(self.weights_complex)
        )  # NCCL does not support complex numbers

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Sensor positions of shape (batch_size, x_dim, num_sensors...).
            u: Input function values of shape (batch_size, u_dim, num_sensors...).
            y: Evaluation coordinates of shape (batch_size, y_dim, num_sensors...).

        Returns:
            Evaluations of the mapped function with shape (batch_size, v_dim, num_sensors...).
        """

        u_fourier = rfft(u, axis=2, norm="forward")

        kernel = torch.view_as_complex(self.kernel)

        out_fourier = torch.einsum(
            "nds,bsn->bdn", kernel, u_fourier[:, : self.num_frequencies, :]
        )

        out = irfft(out_fourier, axis=2, n=y.shape[2], norm="forward")

        return out


class FourierLayer(Operator):
    """Fourier layer. This layer performs an integral kernel operation in Fourier space.

    The convolution with a kernel becomes an element-wise product in Fourier space,
    reducing the complexity of the computation from quadratic to linear.
    For the Fourier transformation pytorch's implementation of the FFT is used.

    Args:
        shapes: Shape of dataset
        num_modes: List with number of modes per fft dimension. The number of
            fft-dimensions is equal to shapes.x.dim. If num_modes is None,
            the maximum number of modes is assumed which is given by the number
            of points per dimension.
        device: Device.

    Example:
        ```python
        dataset = OperatorDataset(x, u, y, v)
        fourier = FourierLayer(dataset.shapes)

        xi, ui, yi, vi = dataset[0:1]
        vi_pred = fourier(xi, ui, yi)
        ```

    """

    TENSOR_INDICES = "defghijklmnopqrstuvxyz"

    def __init__(
        self,
        shapes: OperatorShapes,
        num_modes: Optional[Tuple[int]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(shapes, device)

        assert (
            self.shapes.y.dim == self.shapes.x.dim
        ), f"x.dim == y.dim is a necessary requirement for the FourierLayer. Given y.dim={self.shapes.y.dim} x.dim={self.shapes.x.dim}"

        # make sure self.grid_shape is list
        self.grid_shape = list(self.shapes.u.size)

        self.num_modes = (
            self.grid_shape.copy() if num_modes is None else list(num_modes)
        )
        self.num_modes = [
            min(grid_dim, mode)
            for grid_dim, mode in zip(self.grid_shape, self.num_modes)
        ]

        assert (
            len(self.num_modes) == self.shapes.x.dim
        ), "The number of dimensions specified by 'num_modes' is inconsistent with x.dim."

        # The last dimension is half+1 because we use the `torch.fft.rfftn` method.
        # This is due to the negative frequency modes being redundant.
        self.num_modes[-1] = self.num_modes[-1] // 2 + 1

        weights_shape = (*self.num_modes, shapes.v.dim, shapes.u.dim)

        weights_real = torch.empty(weights_shape, device=device)
        weights_img = torch.empty(weights_shape, device=device)

        nn.init.kaiming_uniform_(weights_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(weights_img, a=math.sqrt(5))

        self.weights_complex = torch.complex(weights_real, weights_img)
        self.kernel = torch.nn.Parameter(
            torch.view_as_real(self.weights_complex)
        )  # NCCL does not support complex numbers

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass. Performs a kernel integral operation in Fourier space.

        Note:
            * `x.dim == y.dim` is a necessary condition for the FourierLayer.
            * `x` and `y` have to be sampled on a regular grid.

        Args:
            x: Sensor positions of shape (batch_size, x_dim, num_sensors...).
            u: Input function values of shape (batch_size, u_dim, num_sensors...).
            y: Evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).

        Returns:
            Evaluations of the mapped function with shape (batch_size, v_dim, num_evaluations...).
        """
        # shapes which can change for different forward passes
        batch_size = y.shape[0]

        # fft related parameters
        num_fft_dimensions = self.shapes.x.dim
        fft_dimensions = list(range(1, num_fft_dimensions + 1))

        # prepare for FFT
        u = u.permute(0, *range(2, u.dim()), 1)
        assert u.dim() == num_fft_dimensions + 2

        # compute n-dimensional real-valued fourier transform
        u_fft = rfftn(u, dim=fft_dimensions, norm="forward")

        # transform Fourier modes from 'standard order' to 'ascending order'
        u_fft = self._get_ascending_order(u_fft, dim=fft_dimensions)

        # add or remove frequencies such that the the fft dimensions of u_fft match self.num_modes
        u_fft = self._add_or_remove_frequencies(
            u_fft, target_shape=self.num_modes, dim=fft_dimensions
        )

        # perform kernel integral operation in Fourier space
        out_fft = self._contract_with_kernel(u_fft, dim=fft_dimensions)

        # the output shape is determined by y.shape[2:] (num_evaluations...) and not x.shape[2:] (num_sensors...)
        target_shape = list(y.shape[2:])

        # fft_shape is the same except last dimension, we only need half the frequencies for the last dimension
        fft_shape = target_shape.copy()
        fft_shape[-1] = fft_shape[-1] // 2 + 1

        # add or remove frequencies such that the fft dimensions of out_fft match fft_shape
        out_fft = self._add_or_remove_frequencies(
            out_fft, target_shape=fft_shape, dim=fft_dimensions
        )

        # transform Fourier modes from 'ascending order' to 'normal order'
        out_fft = self._get_standard_order(out_fft, dim=fft_dimensions)

        # transform back into real-space
        out = irfftn(
            out_fft,
            dim=fft_dimensions,
            s=target_shape,
            norm="forward",
        )

        # match (batch_size, v_dim, num_evaluations...)
        out = out.permute(0, -1, *range(1, out.dim() - 1))
        assert out.shape == (batch_size, self.shapes.v.dim, *y.size()[2:])

        return out

    def _contract_with_kernel(
        self, fft_values: torch.Tensor, dim: Tuple[int]
    ) -> torch.Tensor:
        """Contract kernel with input values.

        Args:
            fft_values: Tensor with fft values.
            dim: List of fft dimensions.

        Returns:
            Output tensor with shape = (batch_size, ..., v.dim)
        """

        num_fft_dimensions = len(dim)

        assert (
            len(self.TENSOR_INDICES) > num_fft_dimensions
        ), f"Too many dimensions. The current limit for the number of dimensions is {len(self.TENSOR_INDICES)}."

        # contraction equation for torch.einsum method
        # d: v-dim, s: u-dim, b: batch-dim
        frequency_indices = "".join(self.TENSOR_INDICES[: int(num_fft_dimensions)])
        contraction_equation = "{}ac,b{}c->b{}a".format(
            frequency_indices, frequency_indices, frequency_indices
        )

        # perform kernel operation in Fourier space
        kernel = torch.view_as_complex(self.kernel)
        out_fft = torch.einsum(contraction_equation, kernel, fft_values)

        return out_fft

    def _remove_large_frequencies(
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
            target_shape:  Determines output shape of fft-dimensions. Should
                be same length as `dim`.

        Returns:
            Tensor with reduced number of fft dimensions.
        """

        assert len(dim) == len(target_shape)

        # by default: don't slice tensor. `slice(None)` is equivalent to `[:]`
        slices = [slice(None)] * fft_values.dim()

        # loop over fft dimensions except last dimension
        for idx, dimension in enumerate(dim[:-1]):
            num_modes_input = fft_values.shape[dimension]

            # if input is already smaller, don't do anything
            if num_modes_input <= target_shape[idx]:
                slices[dimension] = slice(None)
                continue

            center = num_modes_input // 2
            upper = center + target_shape[idx] // 2 + target_shape[idx] % 2
            lower = center - target_shape[idx] // 2
            slices[dimension] = slice(lower, upper)

        # one sided slicing for last dimension
        slices[dim[-1]] = slice(target_shape[-1])
        u_fourier_sliced = fft_values[slices]

        return u_fourier_sliced

    def _get_ascending_order(
        self, fft_values: torch.Tensor, dim: Tuple[int]
    ) -> torch.Tensor:
        """Reorder fft values from 'standard order' to 'ascending order'.

        E.g.:
            Standard order (0, -1, -2, 2, 1) -> Ascending order (-2, -1, 0, 1, 2)

        Note: Last dimension is skipped because when using the real-valued fft,
        `torch.fft.rfftn`, the last dimension only contains positive frequency
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

    def _get_standard_order(
        self, fft_values: torch.Tensor, dim: Tuple[int]
    ) -> torch.Tensor:
        """Reorder fft values from 'ascending order' to 'standard order'.

        E.g.:
            Ascending order (-2, -1, 0, 1, 2) -> Standard order (0, -1, -2, 2, 1)

        Note: Last dimension is skipped because when using the real-valued fft,
        `torch.fft.rfftn`, the last dimension only contains positive frequency
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

    def _zero_padding(
        self, fft_values: torch.Tensor, target_shape: Tuple[int], dim: Tuple[int]
    ) -> torch.Tensor:
        """Add zeros at large positive and negative frequency positions.

        This method assumes that the input tensor is in 'ascending order'.
        See `get_ascending_order` for more details.

        Example:
            fft_values.shape -> (10, 32, 17, 2)
            out = zero_padding(fft_values, dim=(1, 2), target_shape=(30, 16))
            out.shape -> (10, 30, 16, 2)

        Args:
            fft_values: tensor with fft values.
            dim: List of fft dimensions.
            target_shape: Determines output shape of fft-dimensions. Should
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

    def _add_or_remove_frequencies(
        self, fft_values: torch.Tensor, target_shape: Tuple[int], dim: Tuple[int]
    ) -> torch.Tensor:
        """Add or remove frequencies depending on the target_shape parameter.

        If the dimensions in `target_shape` are larger than the fft dimensions,
            zeros are added as large positive and negative frequencies.
        If the dimensions in `target_shape` are smaller than the fft dimensions,
            large positive and negative frequencies are removed.

        This method assumes that the input tensor is in 'ascending order'.
        See `get_ascending_order` for more details.

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
        fft_values = self._remove_large_frequencies(
            fft_values, target_shape=target_shape, dim=dim
        )

        # add zero-frequencies if num_sensors_per_dimension < self.num_modes
        fft_values = self._zero_padding(fft_values, target_shape=target_shape, dim=dim)

        return fft_values
