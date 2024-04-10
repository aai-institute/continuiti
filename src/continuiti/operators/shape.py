"""
`continuiti.operators.shape`
"""

from dataclasses import dataclass


@dataclass
class TensorShape:
    num: int  # number of separate instances
    dim: int  # dimensionality of a single instance (needs to be flat)


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
