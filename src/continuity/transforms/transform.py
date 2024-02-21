import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class Transform(nn.Module, ABC):
    """Abstract base class for transformations of tensors.

    Transformations are applied to tensors to improve model performance, enhance generalization, handle varied input
    sizes, facilitate specific features, reduce overfitting, improve computational efficiency or many other reasons.
    This class takes some tensor and transforms it into some other tensor.
    """

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args: Arguments passed to nn.Module parent class.
            **kwargs: Arbitrary keyword arguments passed to nn.Module parent class.
        """
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the transformation.

        Args:
            tensor: Tensor that should be transformed.

        Returns:
            Transformed tensor.
        """

    @abstractmethod
    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        """Undoes the inverse (given the transformation is bijective).

        When the transformation is not bijective (one-to-one correspondence of data), the inverse/backward
        transformation is not applied. Instead, a warning should be given to the user and an appropriate approximate
        inverse transformation should be provided.

        Args:
            tensor: Transformed tensor.

        Returns:
            Tensor with the transformation undone (given it is possible).
        """
