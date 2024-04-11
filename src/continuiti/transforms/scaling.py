"""
`continuiti.transforms.scaling`
"""

import torch
import torch.nn as nn
from .transform import Transform


class Normalize(Transform):
    r"""Normalization transformation (Z-normalization).

    This transformation takes a mean $\mu$ and standard deviation $\sigma$
    to scale tensors $x$ according to

    $$\operatorname{Normalize}(x) = \frac{x - \mu}{\sigma + \varepsilon} := z,$$

    where $\varepsilon$ is a small value to prevent division by zero.

    Attributes:
        epsilon: small value to prevent division by zero (`torch.finfo.tiny`)

    Args:
        mean: mean used to scale tensors
        std: standard deviation used to scale tensors
    """

    epsilon = torch.finfo(torch.get_default_dtype()).tiny

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.std = nn.Parameter(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply normalization to the input tensor.

        $$z = \frac{x - \mu}{\sigma + \varepsilon}$$

        Args:
            x: input tensor $x$

        Returns:
            normalized tensor $z$
        """
        return (x - self.mean) / (self.std + self.epsilon)

    def undo(self, z: torch.Tensor) -> torch.Tensor:
        r"""Undo the normalization.

        $$x = z~(\sigma + \varepsilon) + \mu$$

        Args:
            z: (normalized) tensor $z$

        Returns:
            un-normalized tensor $x$
        """
        return z * (self.std + self.epsilon) + self.mean
