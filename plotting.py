import numpy as np
import matplotlib.pyplot as plt


def plot_observation(observation, ax=plt.gca()):
    """Plot observation"""
    x = [s.x for s in observation.sensors]
    u = [s.u for s in observation.sensors]
    ax.plot(x, u, 'ko')


def plot_evaluation(model, dataset, observation, ax=plt.gca()):
    """Plot evaluation"""
    n = 100
    x = np.linspace(-1, 1, n)
    u = np.zeros_like(x)
    for i in range(n):
        obs_pos = dataset.flatten(observation, np.array([x[i]]))
        obs_pos = obs_pos.reshape(1, -1)
        u[i] = model(obs_pos)
    ax.plot(x, u, 'k-')