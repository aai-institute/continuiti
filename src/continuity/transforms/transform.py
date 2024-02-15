import torch
import torch.nn as nn
import warnings

from abc import ABC, abstractmethod


class Transform(nn.Module, ABC):
    """Abstract base class for transformations of tensors.

    Transformations are applied to tensors to improve model performance, enhance generalization, handle varied input
    sizes, facilitate specific features, reduce overfitting, improve computational efficiency or many other reasons.
    This class takes some tensor and transforms it into some other tensor.
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

    def backward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the inverse transformation (given the transformation is bijective).

        When the transformation is not bijective (one-to-one correspondence of data) the inverse/backward transformation
        is not applied. Instead, a warning is raised.

        Args:
            tensor: Transformed tensor.

        Returns:
            Tensor with the transformation undone (given it is possible).
        """
        warnings.warn(
            f"Backward pass for transformation {self.__class__.__name__} not implement! "
            f"Backward pass is performed as identity!",
            stacklevel=2,
        )
        return tensor
