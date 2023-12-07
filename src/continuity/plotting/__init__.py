"""Plotting utilities for Continuity."""

import torch
import numpy as np
from typing import Optional
from matplotlib.axis import Axis
import matplotlib.pyplot as plt
from continuity.data import device, Observation
from continuity.model import Operator


def plot_observation(observation: Observation, ax: Optional[Axis] = None):
    """Plots an observation.

    Currently only supports coordinate dimensions of $d = 1,2$.

    Args:
        observation: Observation object
        ax: Axis object. If None, `plt.gca()` is used.
    """
    if ax is None:
        ax = plt.gca()

    x = [s.x for s in observation.sensors]
    u = [s.u for s in observation.sensors]

    dim = x[0].shape[0]
    assert dim in [1, 2], "Only supports `d = 1,2`"

    if dim == 1:
        ax.plot(x, u, "k.")

    if dim == 2:
        xx, yy = [x[0] for x in x], [x[1] for x in x]
        ax.scatter(xx, yy, s=20, c=u, cmap="jet")


def plot_evaluation(
    operator: Operator, observation: Observation, ax: Optional[Axis] = None
):
    """Plots the mapped function `operator(observation)` evaluated on a $[-1, 1]^d$ grid.

    Currently only supports coordinate dimensions of $d = 1,2$.

    Args:
        operator: Operator object
        observation: Observation object
        ax: Axis object. If None, `plt.gca()` is used.
    """
    if ax is None:
        ax = plt.gca()

    dim = observation.coordinate_dim
    assert dim in [1, 2], "Only supports `d = 1,2`"

    if dim == 1:
        n = 200
        y = torch.linspace(-1, 1, n, device=device).reshape(1, -1, 1)
        xu = observation.to_tensor().unsqueeze(0).to(device)
        v = operator(xu, y).detach()
        ax.plot(y.cpu().flatten(), v.cpu().flatten(), "k-")

    if dim == 2:
        n = 128
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, y)
        u = observation.to_tensor().unsqueeze(0).to(device)
        x = (
            torch.tensor(
                np.array(
                    [np.array([xx[i, j], yy[i, j]]) for i in range(n) for j in range(n)]
                ),
                dtype=u.dtype,
            )
            .unsqueeze(0)
            .to(device)
        )
        u = operator(u, x).detach().cpu()
        u = np.reshape(u, (n, n))
        ax.contourf(xx, yy, u, cmap="jet", levels=100)
        ax.set_aspect("equal")
