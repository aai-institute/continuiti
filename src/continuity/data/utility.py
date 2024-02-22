"""
`continuity.data.utility`

Utility functions for data handling.
"""

import torch


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

    return torch.utils.data.random_split(
        dataset,
        [split, 1 - split],
        generator=generator,
    )


def dataset_loss(dataset, operator, loss_fn):
    """Evaluate operator performance on data set.

    Args:
        dataset: Data set.
        operator: Operator.
        loss_fn: Loss function.
    """
    loss = 0.0

    for x, u, y, v in dataset:
        x, u, y, v = x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0), v.unsqueeze(0)
        loss += loss_fn(operator, x, u, y, v)

    return loss
