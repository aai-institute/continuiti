import torch
import io
import numpy as np
import matplotlib.pyplot as plt
from model import device


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
        x = torch.linspace(-1, 1, n, device=device).reshape(1, -1, 1)
        u = observation.to_tensor().unsqueeze(0).to(device)
        v = model(u, x).detach()
        ax.plot(x.cpu().flatten(), v.cpu().flatten(), 'k-')

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


def plot_to_tensorboard(writer, name="image", step=0):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = plt.imread(buf)
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image.astype(np.float32))
    writer.add_image(name, image, step)
