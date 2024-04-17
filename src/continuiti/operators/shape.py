"""
`continuiti.operators.shape`
"""

import torch
from dataclasses import dataclass


@dataclass
class TensorShape:
    dim: int  # dimensionality of a single instance (needs to be flat)
    size: torch.Size  # size of separate instances, e.g., `100` or `(64, 64)`


@dataclass
class OperatorShapes:
    """Shape of input and output functions of an Operator.

    Attributes:
        x: Sensor locations.
        u: Input function evaluated at sensor locations.
        y: Evaluation locations.
        v: Output function evaluated at evaluation locations.

    """

    x: TensorShape
    u: TensorShape
    y: TensorShape
    v: TensorShape
