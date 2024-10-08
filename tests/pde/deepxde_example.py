import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)


def deepxde_example():
    """Physics-informed DeepONet for Poisson equation in 1D.
    Example from DeepXDE.
    https://deepxde.readthedocs.io/en/latest/demos/operator/poisson.1d.pideeponet.html
    """

    # deepxde sets the device context internally, which can conflict with the testing setup, when dealing with different
    # devices (i.e. GPU and CPU). To ensure that the correct device is set the dependency is isolated.
    import deepxde as dde  # noqa

    # Poisson equation: -u_xx = f
    def equation(x, y, f):
        dy_xx = dde.grad.hessian(y, x)
        return -dy_xx - f

    # Domain is interval [0, 1]
    geom = dde.geometry.Interval(0, 1)

    # Zero Dirichlet BC
    def u_boundary(_):
        return 0

    def boundary(_, on_boundary):
        return on_boundary

    bc = dde.icbc.DirichletBC(geom, u_boundary, boundary)

    # Define PDE
    pde = dde.data.PDE(geom, equation, bc, num_domain=100, num_boundary=2)

    # Function space for f(x) are polynomials
    degree = 3
    space = dde.data.PowerSeries(N=degree + 1)

    # Choose evaluation points
    num_eval_points = 10
    evaluation_points = geom.uniform_points(num_eval_points, boundary=True)

    # Define PDE operator
    pde_op = dde.data.PDEOperatorCartesianProd(
        pde,
        space,
        evaluation_points,
        num_function=100,
    )

    # Setup DeepONet
    dim_x = 1
    p = 32
    net = dde.nn.DeepONetCartesianProd(
        [num_eval_points, 32, p],
        [dim_x, 32, p],
        activation="tanh",
        kernel_initializer="Glorot normal",
    )
    print("Params:", sum(p.numel() for p in net.parameters()))

    # Define and train model
    model = dde.Model(pde_op, net)
    model.compile("adam", lr=0.001)
    model.train(epochs=1000)

    # Plot realizations of f(x)
    n = 3
    features = space.random(n)
    fx = space.eval_batch(features, evaluation_points)

    x = geom.uniform_points(100, boundary=True)
    y = model.predict((fx, x))

    # Setup figure
    fig = plt.figure(figsize=(7, 8))
    plt.subplot(2, 1, 1)
    plt.title("Poisson equation: Source term f(x) and solution u(x)")
    plt.ylabel("f(x)")
    z = np.zeros_like(x)
    plt.plot(x, z, "k-", alpha=0.1)

    # Plot source term f(x)
    for i in range(n):
        plt.plot(evaluation_points, fx[i], ".")

    # Plot solution u(x)
    plt.subplot(2, 1, 2)
    plt.ylabel("u(x)")
    plt.plot(x, z, "k-", alpha=0.1)
    for i in range(n):
        plt.plot(x, y[i], "-")
    plt.xlabel("x")

    plt.show()


if __name__ == "__main__":
    deepxde_example()
