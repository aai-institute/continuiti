"""
`continuiti.transforms.transform`

Transform base class.
"""
import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class Transform(nn.Module, ABC):
    """Abstract base class for transformations of tensors.

    Transformations are applied to tensors to improve model performance, enhance generalization, handle varied input
    sizes, facilitate specific features, reduce overfitting, improve computational efficiency or many other reasons.
    This class takes some tensor and transforms it into some other tensor.

    Args:
        *args: Arguments passed to nn.Module parent class.
        **kwargs: Arbitrary keyword arguments passed to nn.Module parent class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the transformation.

        Args:
            tensor: Tensor that should be transformed.

        Returns:
            Transformed tensor.
        """

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the inverse of the transformation (if it exists).

        Args:
            tensor: Transformed tensor.

        Returns:
            Tensor with the transformation undone.

        Raises:
            NotImplementedError: If the inverse of the transformation is not implemented.
        """
        raise NotImplementedError(
            "The undo method is not implemented for this transform."
        )
