from pinn_benchmark.test_cases.problem import Problem
from time import time
import numpy as np
import deepxde as dde
import torch

torch.manual_seed(0)


def solve_deepxde(problem: Problem, neurons_per_layer=10, quadrature_points=10):
    assert problem.domain.dim == 2
    num_hidden_layers = 2

    geom = dde.geometry.Rectangle(*problem.domain.vertices)

    # Use output transform to impose zero boundary conditions
    def transform(x, u):
        res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        return res * u

    n = quadrature_points
    data = dde.data.PDE(
        geom,
        problem.pde,
        [],
        num_domain=n**2,
        num_boundary=4 * n,
        num_test=(2 * n) ** 2,
    )

    net = dde.maps.FNN(
        [problem.domain.dim] + [neurons_per_layer] * num_hidden_layers + [1],
        "tanh",
        "Glorot uniform",
    )
    net.apply_output_transform(transform)

    model = dde.Model(data, net)
    dde.optimizers.set_LBFGS_options(maxiter=1)
    model.compile("L-BFGS")
    early_stopping = dde.callbacks.EarlyStopping(
        min_delta=1e-14, monitor="loss_test", patience=100
    )

    elapsed = -time()
    model.train(callbacks=[early_stopping])
    elapsed += time()

    n_error = 10 * n
    xs = np.linspace(0, 1, n_error)
    ys = np.linspace(0, 1, n_error)
    x, y = np.meshgrid(xs, ys)

    xy = np.vstack((x.flatten(), y.flatten())).T
    z = model.predict(xy)

    z_true = np.array([problem.exact_solution(x) for x in xy])

    u_pred = z.reshape(x.shape)
    u_true = z_true.reshape(x.shape)

    plot_solution = True
    if plot_solution:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        c1 = ax1.contourf(x, y, u_pred)
        c2 = ax2.contourf(x, y, u_true)
        fig.colorbar(c1, ax=ax1)
        fig.colorbar(c2, ax=ax2)
        ax1.set_title("PINN")
        ax2.set_title("Exact")
        ax1.set_aspect("equal", "box")
        ax2.set_aspect("equal", "box")
        plt.show()

    error = np.linalg.norm(u_pred - u_true) / n_error**2

    print(f"L2-Error: {error:.6e}")
    print(f"Elapsed time: {elapsed:.3e} seconds")

    return error, elapsed
