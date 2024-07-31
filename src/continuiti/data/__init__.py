"""
`continuiti.data`

Data sets in continuiti.
Every data set is a list of `(x, u, y, v)` tuples.
"""

from .dataset import OperatorDataset, MaskedOperatorDataset
from .utility import split, dataset_loss

__all__ = [
    "OperatorDataset",
    "MaskedOperatorDataset",
    "split",
    "dataset_loss",
]
