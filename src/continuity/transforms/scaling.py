import torch
import warnings

from .transform import Transform


class Normalization(Transform):
    r"""Z-normalization transformation.

    This transformation uses the mean $\mu$ and the standard deviation $\sigma$ passed in the initialization and scales
    tensors $x$ with the mapping $z(x)=\frac{x - \mu}{\sigma}$.

    Attributes:
        mean: mean value of the tensor.
        std: standard deviation of the tensor.
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
            mean: mean used to scale tensors in forward. To scale a (100, 40, 3) tensor along the last
            dimension, provide the mean along the last dimension in the shape (1, 1, 3).
            std: standard deviation used to scale tensors in forward. Should have the same shape as the mean.
            epsilon: Value for numerical stability (to prevent divide by zero).
        """
        assert mean.shape == std.shape
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
        r"""
        $$\frac{x - \mu}{\sigma}$$

        Args:
            tensor: input.

        Returns:
            Z-normalized tensor.
        """
        return (tensor - self.mean) / (self.std + self.epsilon)

    def backward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.std + self.epsilon) + self.mean
