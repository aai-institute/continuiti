"""
`continuity.data`

Data sets in Continuity.
Every data set is a list of `(x, u, y, v)` tuples.
"""

from .dataset import OperatorDataset
from .shape import DatasetShapes
from .utility import get_device, split, dataset_loss

__all__ = [
    "OperatorDataset",
    "DatasetShapes",
    "get_device",
    "split",
    "dataset_loss",
]

device = get_device()
