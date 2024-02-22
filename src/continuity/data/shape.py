"""
`continuity.data.shape`
"""

from dataclasses import dataclass


@dataclass
class TensorShape:
    num: int  # number of separate instances
    dim: int  # dimensionality of a single instance (needs to be flat)


@dataclass
class DatasetShapes:
    """Shapes of all elements inside an OperatorDataset.

    Attributes:
        num_observations: Total number of all observations in the dataset.
        x: Sensor locations.
        u: Input function evaluated at sensor locations.
        y: Evaluation locations.
        v: Output function evaluated at evaluation locations.

    """

    num_observations: int
    x: TensorShape
    u: TensorShape
    y: TensorShape
    v: TensorShape
