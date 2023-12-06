import torch
import numpy as np
import matplotlib.pyplot as plt
from continuity.data import device


def plot_observation(observation, ax=None):
    """Plot observation"""
    if ax is None:
        ax = plt.gca()

    x = [s.x for s in observation.sensors]
    u = [s.u for s in observation.sensors]

    dim = x[0].shape[0]
    if dim == 1:
        ax.plot(x, u, "k.")
    if dim == 2:
        xx, yy = [x[0] for x in x], [x[1] for x in x]
        ax.scatter(xx, yy, s=20, c=u, cmap="jet")


def plot_evaluation(model, observation, ax=None):
    """Plot evaluation"""
    if ax is None:
        ax = plt.gca()

    dim = observation.coordinate_dim

    if dim == 1:
        n = 200
        x = torch.linspace(-1, 1, n, device=device).reshape(1, -1, 1)
        u = observation.to_tensor().unsqueeze(0).to(device)
        v = model(u, x).detach()
        ax.plot(x.cpu().flatten(), v.cpu().flatten(), "k-")

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
        u = model(u, x).detach().cpu()
        u = np.reshape(u, (n, n))
        ax.contourf(xx, yy, u, cmap="jet", levels=100)
        ax.set_aspect("equal")
