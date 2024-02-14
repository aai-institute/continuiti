"""
`continuity.plotting`

Plotting utilities for Continuity.
"""

import torch
import numpy as np
from typing import Optional
from matplotlib.axis import Axis
import matplotlib.pyplot as plt
from continuity.operators import Operator


def plot(x: torch.Tensor, u: torch.Tensor, ax: Optional[Axis] = None):
    """Plots a function $u(x)$.

    Currently only supports coordinate dimensions of $d = 1,2$.

    Args:
        x: Spatial coordinates of shape (n, d)
        u: Function values of shape (n, m)
        ax: Axis object. If None, `plt.gca()` is used.
    """
    if ax is None:
        ax = plt.gca()

    dim = x.shape[-1]
    assert dim in [1, 2], "Only supports `d = 1,2`"

    # Move to cpu
    x = x.cpu().detach().numpy()
    u = u.cpu().detach().numpy()

    if dim == 1:
        ax.plot(x, u, ".")

    if dim == 2:
        xx, yy = x[:, 0], x[:, 1]
        s = 50000 / len(xx)
        ax.scatter(xx, yy, marker="s", s=s, c=u, cmap="jet")
        ax.set_aspect("equal")


def plot_evaluation(
    operator: Operator, x: torch.Tensor, u: torch.Tensor, ax: Optional[Axis] = None
):
    """Plots the mapped function `operator(observation)` evaluated on a $[-1, 1]^d$ grid.

    Currently only supports coordinate dimensions of $d = 1,2$.

    Args:
        operator: Operator object
        x: Collocation points of shape (n, d)
        u: Function values of shape (n, c)
        ax: Axis object. If None, `plt.gca()` is used.
    """
    if ax is None:
        ax = plt.gca()

    dim = x.shape[-1]
    assert dim in [1, 2], "Only supports `d = 1,2`"

    if dim == 1:
        n = 200
        y = torch.linspace(-1, 1, n).unsqueeze(-1)
        x = x.unsqueeze(0)
        u = u.unsqueeze(0)
        y = y.unsqueeze(0)
        v = operator(x, u, y).detach()
        ax.plot(y.cpu().flatten(), v.cpu().flatten(), "k-")

    if dim == 2:
        n = 128
        a = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(a, a)
        y = torch.tensor(
            np.array(
                [np.array([xx[i, j], yy[i, j]]) for i in range(n) for j in range(n)]
            ),
            dtype=u.dtype,
        ).unsqueeze(0)
        x = x.unsqueeze(0)
        u = u.unsqueeze(0)
        y = y.unsqueeze(0)
        u = operator(x, u, y).detach().cpu()
        u = np.reshape(u, (n, n))
        ax.contourf(xx, yy, u, cmap="jet", levels=100)
        ax.set_aspect("equal")
