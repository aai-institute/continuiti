import pytest
import torch
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import continuiti as cti

torch.manual_seed(0)


@pytest.mark.slow
def test_pideeponet():
    """Physics-informed DeepONet for Poisson equation in 1D.
    Example from DeepXDE in *continuiti*.
    https://deepxde.readthedocs.io/en/latest/demos/operator/poisson.1d.pideeponet.html
    """

    # Poisson equation: -v_xx = f
    mse = torch.nn.MSELoss()

    def equation(_, f, y, v):
        # PDE
        dy_xx = dde.grad.hessian(v, y)
        inner_loss = mse(-dy_xx, f)

        # BC
        y_bnd, v_bnd = y[:, :, bnd_indices], v[:, :, bnd_indices]
        boundary_loss = mse(v_bnd, v_boundary(y_bnd))

        return inner_loss + boundary_loss

    # Domain is interval [0, 1]
    geom = dde.geometry.Interval(0, 1)

    # Zero Dirichlet BC
    def v_boundary(y):
        return torch.zeros_like(y)

    # Sample domain and boundary points
    num_domain = 32
    num_boundary = 2
    x_domain = geom.uniform_points(num_domain)
    x_bnd = geom.uniform_boundary_points(num_boundary)

    x = np.concatenate([x_domain, x_bnd])
    num_points = len(x)
    bnd_indices = range(num_domain, num_points)

    # Function space for f(x) are polynomials
    degree = 3
    space = dde.data.PowerSeries(N=degree + 1)

    num_functions = 10
    coeffs = space.random(num_functions)
    fx = space.eval_batch(coeffs, x)

    # Specify dataset
    xt = torch.tensor(x.T).requires_grad_(True)
    x_all = xt.expand(num_functions, -1, -1)  # (num_functions, x_dim, num_domain)
    u_all = torch.tensor(fx).unsqueeze(1)  # (num_functions, u_dim, num_domain)
    y_all = x_all  # (num_functions, y_dim, num_domain)
    v_all = torch.zeros_like(y_all)  # (num_functions, v_dim, num_domain)

    dataset = cti.data.OperatorDataset(
        x=x_all,
        u=u_all,
        y=y_all,  # same as x_all
        v=v_all,  # only for shapes
    )

    # Define operator
    operator = cti.operators.DeepONet(
        dataset.shapes,
        trunk_depth=1,
        branch_depth=1,
        basis_functions=32,
    )
    # or any other operator, e.g.:
    # operator = cti.operators.DeepNeuralOperator(dataset.shapes)

    # Define and train model
    loss_fn = cti.pde.PhysicsInformedLoss(equation)
    trainer = cti.Trainer(operator, loss_fn=loss_fn)
    trainer.fit(dataset, epochs=100)

    # Plot realizations of f(x)
    n = 3
    features = space.random(n)
    fx = space.eval_batch(features, x)

    y = geom.uniform_points(200, boundary=True)

    x_plot = torch.tensor(x_domain.T).expand(n, -1, -1)
    u_plot = torch.tensor(fx).unsqueeze(1)
    y_plot = torch.tensor(y.T).expand(n, -1, -1)
    v = operator(x_plot, u_plot, y_plot)
    v = v.detach().numpy()

    fig = plt.figure(figsize=(7, 8))
    plt.subplot(2, 1, 1)
    plt.title("Poisson equation: Source term f(x) and solution v(x)")
    plt.ylabel("f(x)")
    z = np.zeros_like(y)
    plt.plot(y, z, "k-", alpha=0.1)

    # Plot source term f(x)
    for i in range(n):
        plt.plot(x, fx[i], ".")

    # Plot solution v(x)
    plt.subplot(2, 1, 2)
    plt.ylabel("v(x)")
    plt.plot(y, z, "k-", alpha=0.1)
    for i in range(n):
        plt.plot(y, v[i].T, "-")
    plt.xlabel("x")

    plt.savefig("pideeponet.png", dpi=500)


if __name__ == "__main__":
    test_pideeponet()
