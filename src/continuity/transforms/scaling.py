import torch
import warnings

from .transform import Transform


class Normalize(Transform):
    r"""Z-normalization transformation.

    This transformation uses the mean $\mu$ and the standard deviation $\sigma$ passed in the initialization and scales
    tensors $x$ with the mapping $z(x)=\frac{x - \mu}{\sigma}$.

    Attributes:
        mean: mean value used to scale tensors.
        std: standard deviation used to scale tensors.
        epsilon: Value to prevent divide by zero.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        epsilon: float = None,
    ):
        """

        Args:
            mean: mean used to scale tensors.
            std: standard deviation used to scale tensors in forward.
            epsilon: Value for numerical stability (to prevent divide by zero).
        """
        assert torch.all(torch.greater_equal(std, torch.zeros(std.shape)))
        super().__init__()
        self.mean = mean
        if torch.allclose(std, torch.zeros(std.shape)):
            warnings.warn(
                "Z-normalization with standard deviation 0! "
                "The feature vector does not have any discriminative power!",
                stacklevel=2,
            )
        self.std = std
        if epsilon is None:
            self.epsilon = torch.finfo(torch.get_default_dtype()).tiny
        else:
            self.epsilon = epsilon

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Applies normalization to the input tensor.

        Args:
            tensor: input.

        Returns:
            normalized tensor.
        """
        return (tensor - self.mean) / (self.std + self.epsilon)

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Reverse the normalization.

        Args:
            tensor: normalized tensor.

        Returns:
            tensor with normalization undone.
        """
        return tensor * (self.std + self.epsilon) + self.mean
