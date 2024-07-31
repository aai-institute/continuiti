"""
`continuiti.data.dataset`

Data sets in continuiti.
Every data set is a list of `(x, u, y, v)` tuples.
"""

import torch
import torch.utils.data as td
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, List, Union
from abc import ABC
from continuiti.transforms import Transform
from continuiti.operators.shape import OperatorShapes, TensorShape


class OperatorDatasetBase(td.Dataset, ABC):
    """Abstract base class of a dataset for operator training."""

    def __init__(self, shapes: OperatorShapes, n_observations: int) -> None:
        super().__init__()
        self.shapes = shapes
        self.n_observations = n_observations

    def _apply_transformations(
        self, src: List[Tuple[torch.Tensor, Optional[Transform]]]
    ) -> List[torch.Tensor]:
        """Applies class transformations to four tensors.

        Args:
            src:

        Returns:
            Input src with class transformations applied.
        """
        out = []
        for src_tensor, transformation in src:
            if transformation is None:
                out.append(src_tensor)
                continue
            out.append(transformation(src_tensor))
        return out

    def __len__(self) -> int:
        """Return the number of observations in the dataset.

        Returns:
            Number of observations in the entire dataset.
        """
        return self.n_observations


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

        self.x = x
        self.u = u
        self.y = y
        self.v = v

        # used to initialize architectures
        shapes = OperatorShapes(
            x=TensorShape(dim=x_dim, size=x_size),
            u=TensorShape(dim=u_dim, size=u_size),
            y=TensorShape(dim=y_dim, size=y_size),
            v=TensorShape(dim=v_dim, size=v_size),
        )

        super().__init__(shapes, len(x))

        self.x_transform = x_transform
        self.u_transform = u_transform
        self.y_transform = y_transform
        self.v_transform = v_transform

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
        tensors = self._apply_transformations(
            [
                (self.x[idx], self.x_transform),
                (self.u[idx], self.u_transform),
                (self.y[idx], self.y_transform),
                (self.v[idx], self.v_transform),
            ]
        )

        return tensors[0], tensors[1], tensors[2], tensors[3]


class MaskedOperatorDataset(OperatorDatasetBase):
    """A dataset for operator training containing masks in addition to tensors describing the mapping.

    Data, especially described on unstructured grids, can vary in the number of evaluations or sensors. Even
    measurements of phenomena do not always contain the same number of sensors and or evaluations. This dataset is able
    to handle datasets that have differing number of sensors or evaluations. For this masks, both for the input and
    output space, describe which values are relevant to the dataset and which are irrelevant padding values, that
    should be ignored during training and evaluation. Padding tensors is important to efficiently handle data in
    batches.

    Args:
        x: Tensor of shape (num_observations, x_dim, num_sensors) with sensor positions or list containing
            num_observations tensors of shape (x_dim, ...).
        u: Tensor of shape (num_observations, u_dim, num_sensors) with evaluations of the input functions at sensor
            positions or list containing num_observations tensors of shape (u_dim, ...).
        y: Tensor of shape (num_observations, y_dim, num_evaluations) with evaluation positions or list containing
            num_observations tensors of shape (y_dim, ...).
        v: Tensor of shape (num_observations, v_dim, num_evaluations) with ground truth operator mappings or list
            containing num_observations tensors of shape (v_dim, ...).
        ipt_mask:Boolean tensor of shape (num_observations, num_sensors) with True indicating that a value pair of the
            input space should be taken into consideration during training.
        opt_mask: Boolean tensor of shape (num_observations, num_evaluations) with True indicating that a value pair of
            the output space should be taken into consideration during training.
        x_transform: Transformation applied to x.
        u_transform: Transformation applied to u.
        y_transform: Transformation applied to y.
        v_transform: Transformation applied to v.

    """

    def __init__(
        self,
        x: Union[torch.Tensor, List[torch.Tensor]],
        u: Union[torch.Tensor, List[torch.Tensor]],
        y: Union[torch.Tensor, List[torch.Tensor]],
        v: Union[torch.Tensor, List[torch.Tensor]],
        ipt_mask: Optional[torch.Tensor] = None,
        opt_mask: Optional[torch.Tensor] = None,
        x_transform: Optional[Transform] = None,
        u_transform: Optional[Transform] = None,
        y_transform: Optional[Transform] = None,
        v_transform: Optional[Transform] = None,
    ) -> None:
        assert (
            len(x) == len(u) == len(y) == len(v)
        ), f"All tensors need to have the same number of observations, but found {len(x)}, {len(u)}, {len(y)}, {len(v)}."
        ipt_is_list = isinstance(x, list)
        assert self._is_valid_space(x, u, ipt_mask)
        opt_is_list = isinstance(y, list)
        assert self._is_valid_space(y, v, opt_mask)

        if ipt_is_list:
            x, u, ipt_mask = self._pad_list_space(x, u)

        if opt_is_list:
            y, v, opt_mask = self._pad_list_space(y, v)

        self.x = x
        self.u = u
        self.y = y
        self.v = v

        self.ipt_mask = ipt_mask
        self.opt_mask = opt_mask

        self.x_transform = x_transform
        self.u_transform = u_transform
        self.y_transform = y_transform
        self.v_transform = v_transform

        super().__init__(
            shapes=OperatorShapes(
                x=TensorShape(
                    dim=x[0].size(1), size=torch.Size([])
                ),  # size agnostic dataset
                u=TensorShape(dim=u[0].size(1), size=torch.Size([])),
                y=TensorShape(dim=y[0].size(1), size=torch.Size([])),
                v=TensorShape(dim=v[0].size(1), size=torch.Size([])),
            ),
            n_observations=len(x),
        )

    def _is_valid_space(
        self,
        member: Union[torch.Tensor, List[torch.Tensor]],
        values: Union[torch.Tensor, List[torch.Tensor]],
        mask: Optional[torch.Tensor],
    ) -> bool:
        """Asseses whether a space is in alignment with its respective requirements.

        Depending on whether a space is described by a list of tensors or a tensor certain argument need to align.
        All observations need to have the same dimensions in their respective samples. The domain and the function
        values need to be described by the same number of observations.

        Args:
            member: Tensor of locations.
            values: Function values evaluated in the domain locations.
            mask: Boolean mask where a True value indicates that a specific sample should be taken into consideration.

        Returns:
            A boolean value True when the space description is valid.
        """
        assert type(member) is type(
            values
        ), f"All types of tensors in one space need to match. But found {type(member)} and {type(values)}."

        ndim: int
        if mask is not None:
            assert isinstance(
                member, torch.Tensor
            ), f"When providing a mask the member and values need to be tensors. But found {type(member)}"
            ndim = member.dim() - 1  # remove batch dimension
        else:
            assert all(
                [di.size(0) == member[0].size(0) for di in member[1:]]
            ), "Dimensions of all samples of the member need to match."
            assert all(
                [vi.size(0) == values[0].size(0) for vi in values[1:]]
            ), "Dimensions of all function values need to match."
            ndim = member[0].dim()

        assert (
            ndim == 2
        ), f"{self.__class__.__name__} currently only supports exactly one dim and one size dimension."

        return True

    def _pad_list_space(
        self, member: List[torch.Tensor], values: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transforms a space described by lists of tensors (image) to a space described by padded tensors and a mask.

        Args:
            member: List of tensors describing the locations of the samples.
            values: List of tensors describing the function values of the samples.

        Returns:
            padded member tensor, padded values tensor, and matching mask.
        """
        assert not any(
            [torch.any(torch.isinf(mi)) for mi in member]
        ), "Expects domain to be truncated in finite space."

        member_padded = pad_sequence(
            [mi.transpose(0, 1) for mi in member],
            batch_first=True,
            padding_value=torch.inf,
        ).transpose(1, 2)
        values_padded = pad_sequence(
            [vi.transpose(0, 1) for vi in values], batch_first=True, padding_value=0
        ).transpose(1, 2)

        mask = member_padded != torch.inf
        member_padded[
            ~mask
        ] = 0  # mask often applied by adding a tensor with -inf values in masked locations (e.g. in scaled dot product).

        return member_padded, values_padded, mask

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Retrieves the input-output pair at the specified index and applies transformations.

        Parameters:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the three input tensors, the output tensor, and masks for both the input and output for
                the given index.
        """
        tensors = self._apply_transformations(
            [
                (self.x[idx], self.x_transform),
                (self.u[idx], self.u_transform),
                (self.y[idx], self.y_transform),
                (self.v[idx], self.v_transform),
            ]
        )

        if self.ipt_mask is not None:
            ipt_mask = self.ipt_mask[idx]
        else:
            ipt_mask = None

        if self.opt_mask is not None:
            opt_mask = self.opt_mask[idx]
        else:
            opt_mask = None

        return tensors[0], tensors[1], tensors[2], tensors[3], ipt_mask, opt_mask
