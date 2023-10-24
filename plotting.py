import numpy as np
import matplotlib.pyplot as plt


def plot_observation(observation, ax=plt.gca()):
    """Plot observation"""
    x = [s.x for s in observation.sensors]
    u = [s.u for s in observation.sensors]

    dim = x[0].shape[0]
    if dim == 1:
        ax.plot(x, u, 'k.')
    if dim == 2:
        xx, yy = [x[0] for x in x], [x[1] for x in x]
        ax.scatter(xx, yy, s=20, c=u, cmap='jet')


def plot_evaluation(model, dataset, observation, ax=plt.gca()):
    """Plot evaluation"""
    dim = dataset.coordinate_dim

    if dim == 1:
        n = 200
        x = np.linspace(-1, 1, n)
        u = np.zeros_like(x)
        for i in range(n):
            obs_pos = dataset.flatten(observation, np.array([x[i]]))
            obs_pos = obs_pos.reshape(1, -1)
            u[i] = model(obs_pos)
        ax.plot(x, u, 'k-')

    if dim == 2:
        n = 128
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        xx, yy = np.meshgrid(x, y)
        u = np.zeros_like(xx)
        obs_pos = np.array([
            dataset.flatten(
                observation, 
                np.array([xx[i, j], yy[i, j]])
            )
            for i in range(n)
            for j in range(n)
        ])
        u = model(obs_pos)
        u = np.reshape(u, (n, n))
        ax.contourf(xx, yy, u, cmap='jet', levels=100)
        ax.set_aspect('equal')