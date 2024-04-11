"""
`continuiti.transforms.compose`

Composes multiple transformations.
"""

import torch
from typing import List

from .transform import Transform


class Compose(Transform):
    """Handles the chained sequential application of multiple transformations.

    Args:
        transforms: transformations that should be applied in the order they are in the list.
        *args: Arguments of parent class.
        **kwargs: Arbitrary keyword arguments of parent class.

    Attributes:
        transforms (List): Encapsulates multiple transformations into one.
    """

    def __init__(self, transforms: List[Transform], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies multiple transformations to a tensor in sequential order.

        Args:
            tensor: Tensor to be transformed.

        Returns:
            Tensor with all transformations applied.
        """
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        """Undoes multiple transformations.

        Args:
            tensor: Transformed tensor.

        Returns:
            Tensor with undone transformations (if possible).
        """
        for transform in reversed(self.transforms):
            tensor = transform.undo(tensor)
        return tensor
