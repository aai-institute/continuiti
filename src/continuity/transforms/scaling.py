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

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """

        Args:
            mean: mean used to scale tensors.
            std: standard deviation used to scale tensors in forward.
        """
        super().__init__()
        self.mean = mean
        if not std.all():
            # some value is zero
            epsilon = torch.finfo(torch.get_default_dtype()).tiny
            warnings.warn(
                "Normalization with standard deviation 0. "
                "Some or all features do not have any discriminative power! "
                f"Introducing a epsilon={epsilon} to account for numerical stability.",
                stacklevel=2,
            )
            std[std == 0] = epsilon
        self.std = std

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Applies normalization to the input tensor.

        Args:
            tensor: input.

        Returns:
            normalized tensor.
        """
        return (tensor - self.mean) / self.std

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""Reverse the normalization.

        Args:
            tensor: normalized tensor.

        Returns:
            tensor with normalization undone.
        """
        return tensor * self.std + self.mean
