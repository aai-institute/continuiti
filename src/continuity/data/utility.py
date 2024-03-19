"""
`continuity.data.utility`

Utility functions for data handling.
"""

import torch
from typing import Optional
from continuity.operators.losses import Loss, MSELoss


def split(dataset, split=0.5, seed=None):
    """
    Split data set into two parts.

    Args:
        split: Split fraction.
    """
    assert 0 < split < 1, "Split fraction must be between 0 and 1."

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    size = len(dataset)
    split = int(size * split)

    return torch.utils.data.random_split(
        dataset,
        [split, size - split],
        generator=generator,
    )


def dataset_loss(
    dataset,
    operator,
    loss_fn: Optional[Loss] = None,
    device: Optional[torch.device] = None,
):
    """Evaluate operator performance on data set.

    Args:
        dataset: Data set.
        operator: Operator.
        loss_fn: Loss function. Default is MSELoss.
        device: Device to evaluate on. Default is CPU.
    """
    loss_fn = loss_fn or MSELoss()
    device = device or torch.device("cpu")

    x, u, y, v = dataset[:]
    x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)

    return loss_fn(operator, x, u, y, v).item()
