"""
`continuiti.transforms.quantile_scaler`

Quantile Scaler class.
"""

import torch
import torch.nn as nn
from continuiti.transforms import Transform
from typing import Union, Tuple


class QuantileScaler(Transform):
    """Quantile Scaler Class.

    A transform for scaling input data to a specified target distribution using quantiles. This is
    particularly useful for normalizing data in a way that is more robust to outliers than standard
    z-score normalization.

    The transformation maps the quantiles of the input data to the quantiles of the target distribution,
    effectively performing a non-linear scaling that preserves the relative distribution of the data.

    Args:
        src: tensor from which the source distribution is drawn.
        n_quantile_intervals: Number of individual bins into which the data is categorized.
        target_mean: Mean of the target Gaussian distribution. Can be float (all dimensions use the same mean), or
            tensor (allows for different means along different dimensions).
        target_std: Std of the target Gaussian distribution. Can be float (all dimensions use the same std), or
            tensor (allows for different stds along different dimensions).
        eps: Small value to bound the target distribution to a finite interval.

    """

    def __init__(
        self,
        src: torch.Tensor,
        n_quantile_intervals: int = 1000,
        target_mean: Union[float, torch.Tensor] = 0.0,
        target_std: Union[float, torch.Tensor] = 1.0,
        eps: float = 1e-3,
    ):
        assert eps <= 0.5
        assert eps >= 0

        super().__init__()

        if isinstance(target_mean, float):
            target_mean = target_mean * torch.ones(1)
        if isinstance(target_std, float):
            target_std = target_std * torch.ones(1)

        self.target_mean = target_mean
        self.target_std = target_std

        assert n_quantile_intervals > 0
        self.n_quantile_intervals = n_quantile_intervals
        self.n_q_points = n_quantile_intervals + 2  # n intervals have n + 2 edges

        self.n_dim = src.size(-1)

        # source "distribution"
        self.quantile_fractions = torch.linspace(0, 1, self.n_q_points)
        quantile_points = torch.quantile(
            src.view(-1, self.n_dim),
            self.quantile_fractions,
            dim=0,
            interpolation="linear",
        )
        self.quantile_points = nn.Parameter(quantile_points)
        self.deltas = nn.Parameter(quantile_points[1:] - quantile_points[:-1])

        # target distribution
        self.target_distribution = torch.distributions.normal.Normal(
            target_mean, target_std
        )
        self.target_quantile_fractions = torch.linspace(
            0 + eps, 1 - eps, self.n_q_points
        )  # bounded domain
        target_quantile_points = self.target_distribution.icdf(
            self.target_quantile_fractions
        )
        target_quantile_points = target_quantile_points.unsqueeze(1).repeat(
            1, self.n_dim
        )
        self.target_quantile_points = nn.Parameter(target_quantile_points)
        self.target_deltas = nn.Parameter(
            target_quantile_points[1:] - target_quantile_points[:-1]
        )

    def _get_scaling_indices(
        self, src: torch.Tensor, quantile_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to get the indices of a tensor closest to src.

        Args:
            src: Input tensor.
            quantile_tensor: Tensor containing quantile interval information of a distribution.

        Returns:
            Tuple containing the indices with the same shape as src with indices of quantile_tensor where the distance
                between src and quantile_tensor is minimal, according to the last dim.
        """
        assert src.size(-1) == self.n_dim

        # preprocess tensors
        v1 = src
        v2 = quantile_tensor
        work_ndim = max([v1.ndim, v2.ndim])

        v2_shape = [1] * (work_ndim - v2.ndim) + list(v2.shape)
        v2 = v2.view(*v2_shape)
        v2 = v2.unsqueeze(0)

        v1_shape = [1] * (work_ndim - v1.ndim) + list(v1.shape)
        v1 = v1.view(*v1_shape)
        v1 = v1.unsqueeze(v2.ndim - 2)

        work_dims = torch.Size([max([a, b]) for a, b in zip(v1.shape, v2.shape)])
        v1 = v1.expand(work_dims)
        v2 = v2.expand(work_dims)

        # find left boundary inside quantile intervals
        diff = v2 - v1
        diff[diff >= 0] = -torch.inf  # discard right boundaries
        indices = diff.argmax(dim=-2)  # defaults to zero when all values are -inf
        indices[indices > self.n_quantile_intervals] -= 1  # right boundary overflow

        # prepare for indexing
        return (
            indices.view(-1),
            torch.arange(self.n_dim).repeat(src.nelement() // self.n_dim),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms the input tensor to match the target distribution using quantile scaling.

        Args:
            tensor: The input tensor to transform.

        Returns:
            The transformed tensor, scaled to the target distribution.
        """
        indices = self._get_scaling_indices(tensor, self.quantile_points)
        # Scale input tensor to the unit interval based on source quantiles
        p_min = self.quantile_points[indices].view(tensor.shape)
        delta = self.deltas[indices].view(tensor.shape)
        out = tensor - p_min
        out = out / delta

        # Scale and shift to match the target distribution
        p_t_min = self.target_quantile_points[indices].view(tensor.shape)
        delta_t = self.target_deltas[indices].view(tensor.shape)
        out = out * delta_t
        out = out + p_t_min

        return out

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverses the transformation applied by the forward method, mapping the tensor back to its original
        distribution.

        Args:
            tensor: The tensor to reverse the transformation on.

        Returns:
            The tensor with the quantile scaling transformation reversed according to the src distribution.
        """
        indices = self._get_scaling_indices(tensor, self.target_quantile_points)

        # Scale input tensor to the unit interval based on the target distribution
        p_t_min = self.target_quantile_points[indices].view(tensor.shape)
        delta_t = self.target_deltas[indices].view(tensor.shape)
        out = tensor - p_t_min
        out = out / delta_t

        # Scale and shift to match the src distribution
        p_min = self.quantile_points[indices].view(tensor.shape)
        delta = self.deltas[indices].view(tensor.shape)
        out = out * delta
        out = out + p_min

        return out
