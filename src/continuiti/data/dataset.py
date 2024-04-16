"""
`continuiti.data.dataset`

Data sets in continuiti.
Every data set is a list of `(x, u, y, v)` tuples.
"""

import torch
import torch.utils.data as td
from typing import Optional, Tuple
from abc import ABC, abstractmethod
from continuiti.transforms import Transform
from continuiti.operators.shape import OperatorShapes, TensorShape


class OperatorDatasetBase(td.Dataset, ABC):
    """Abstract base class of a dataset for operator training."""

    shapes: OperatorShapes

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            number of samples in the entire set.
        """

    @abstractmethod
    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves the input-output pair at the specified index and applies transformations.

        Parameters:
            - idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the three input tensors and the output tensor for the given index.
        """


class OperatorDataset(OperatorDatasetBase):
    """A dataset for operator training.

    In operator training, at least one function is mapped onto a second one. To fulfill the properties discretization
    invariance, domain independence and learn operators with physics-based loss access to at least four different
    discretized spaces is necessary. One on which the input is sampled (x), the input function sampled on these points
    (u), the discretization of the output space (y), and the output of the operator (v) sampled on these points. Not
    all loss functions and/or operators need access to all of these attributes.

    Args:
        x: Tensor of shape (num_observations, x_dim, num_sensors...) with sensor positions.
        u: Tensor of shape (num_observations, u_dim, num_sensors...) with evaluations of the input functions at sensor positions.
        y: Tensor of shape (num_observations, y_dim, num_evaluations...) with evaluation positions.
        v: Tensor of shape (num_observations, v_dim, num_evaluations...) with ground truth operator mappings.

    Attributes:
        shapes: Shape of all tensors.
        transform: Transformations for each tensor.
    """

    def __init__(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        x_transform: Optional[Transform] = None,
        u_transform: Optional[Transform] = None,
        y_transform: Optional[Transform] = None,
        v_transform: Optional[Transform] = None,
    ):
        assert all([t.ndim >= 3 for t in [x, u, y, v]]), "Wrong number of dimensions."
        assert (
            x.size(0) == u.size(0) == y.size(0) == v.size(0)
        ), "Inconsistent number of observations."

        # get dimensions and sizes
        x_dim, x_size = x.size(1), x.size()[2:]
        u_dim, u_size = u.size(1), u.size()[2:]
        y_dim, y_size = y.size(1), y.size()[2:]
        v_dim, v_size = v.size(1), v.size()[2:]

        assert x_size == u_size, "Inconsistent number of sensors."
        assert y_size == v_size, "Inconsistent number of evaluations."

        super().__init__()

        self.x = x
        self.u = u
        self.y = y
        self.v = v

        # used to initialize architectures
        self.shapes = OperatorShapes(
            x=TensorShape(dim=x_dim, size=x_size),
            u=TensorShape(dim=u_dim, size=u_size),
            y=TensorShape(dim=y_dim, size=y_size),
            v=TensorShape(dim=v_dim, size=v_size),
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
            Number of samples in the entire set.
        """
        return self.x.size(0)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves the input-output pair at the specified index and applies transformations.

        Parameters:
            idx: The index of the sample to retrieve.

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
            x: Tensor of shape (num_observations, x_dim, num_sensors...) with sensor positions.
            u: Tensor of shape (num_observations, u_dim, num_sensors...) with evaluations of the input functions at sensor positions.
            y: Tensor of shape (num_observations, y_dim, num_evaluations...) with evaluation positions.
            v: Tensor of shape (num_observations, v_dim, num_evaluations...) with ground truth operator mappings.

        Returns:
            Input samples with class transformations applied.
        """
        sample = {"x": x, "u": u, "y": y, "v": v}

        # transform
        for dim, val in sample.items():
            if dim in self.transform:
                sample[dim] = self.transform[dim](val)

        return sample["x"], sample["u"], sample["y"], sample["v"]
