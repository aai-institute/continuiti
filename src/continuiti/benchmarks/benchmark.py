"""
`continuiti.benchmarks.benchmark`

Benchmark base class.
"""

from typing import List
from dataclasses import dataclass, field
from continuiti.data import OperatorDataset
from continuiti.operators.losses import Loss, MSELoss


@dataclass
class Benchmark:
    """Benchmark class.

    A Benchmark object encapsulates two distinct datasets: a train and a test dataset. The training dataset is used to
    train an operator to fit the dataset. The test dataset, in contrast, is utilized solely for evaluating the
    performance. The evaluation is done by measuring the loss on the test set.
    """

    train_dataset: OperatorDataset
    test_dataset: OperatorDataset
    losses: List[Loss] = field(
        default_factory=lambda: [
            MSELoss(),
        ]
    )

    def __str__(self):
        """Return string representation of the benchmark."""
        return self.__class__.__name__
