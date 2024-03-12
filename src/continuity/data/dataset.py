"""
`continuity.data.dataset`

Data sets in Continuity.
Every data set is a list of `(x, u, y, v)` tuples.
"""

import torch
import torch.utils.data as td
from typing import Tuple

from .shape import DatasetShapes, TensorShape


class OperatorDataset(td.Dataset):
    """A dataset for operator training.

    In operator training, at least one function is mapped onto a second one. To fulfill the properties discretization
    invariance, domain independence and learn operators with physics-based loss access to at least four different
    discretized spaces is necessary. One on which the input is sampled (x), the input function sampled on these points
    (u), the discretization of the output space (y), and the output of the operator (v) sampled on these points. Not
    all loss functions and/or operators need access to all of these attributes.

    Args:
        x: Tensor of shape (#observations, #sensors, x-dim) with sensor positions.
        u: Tensor of shape (#observations, #sensors, u-dim) with evaluations of the input functions at sensor positions.
        y: Tensor of shape (#observations, #evaluations, y-dim) with evaluation positions.
        v: Tensor of shape (#observations, #evaluations, v-dim) with ground truth operator mappings.

    Attributes:
        shapes (dataclass): Shape of all tensors.
        transform (dict): Transformations for each tensor.
    """

    def __init__(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        x_transform=None,
        u_transform=None,
        y_transform=None,
        v_transform=None,
    ):
        assert x.ndim == u.ndim == y.ndim == v.ndim == 3, "Wrong number of dimensions."
        assert (
            x.size(0) == u.size(0) == y.size(0) == v.size(0)
        ), "Inconsistent number of observations."
        assert x.size(1) == u.size(1), "Inconsistent number of sensors."
        assert y.size(1) == v.size(1), "Inconsistent number of evaluations."

        super().__init__()

        self.x = x
        self.u = u
        self.y = y
        self.v = v

        # used to initialize architectures
        self.shapes = DatasetShapes(
            num_observations=int(x.size(0)),
            x=TensorShape(*x.size()[1:]),
            u=TensorShape(*u.size()[1:]),
            y=TensorShape(*y.size()[1:]),
            v=TensorShape(*v.size()[1:]),
        )

        self.transform = {
            dim: tf
            for dim, tf in [
                ("x", x_transform),
                ("u", u_transform),
                ("y", y_transform),
                ("v", v_transform),
            ]
            if tf is not None
        }

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            number of samples in the entire set.
        """
        return self.shapes.num_observations

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves the input-output pair at the specified index and applies transformations.

        Parameters:
            - idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the three input tensors and the output tensor for the given index.
        """
        return self._apply_transformations(
            self.x[idx], self.u[idx], self.y[idx], self.v[idx]
        )

    def _apply_transformations(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies class transformations to four tensors.

        Args:
            x: Tensor of shape (#samples, #sensors, x-dim) with sensor positions.
            u: Tensor of shape (#samples, #sensors, u-dim) with evaluations of the input functions at sensor positions.
            y: Tensor of shape (#samples, #evaluations, y-dim) with evaluation positions.
            v: Tensor of shape (#samples, #evaluations, v-dim) with ground truth operator mappings.

        Returns:
            Input samples with class transformations applied.
        """
        sample = {"x": x, "u": u, "y": y, "v": v}

        # transform
        for dim, val in sample.items():
            if dim in self.transform:
                sample[dim] = self.transform[dim](val)

        return sample["x"], sample["u"], sample["y"], sample["v"]
