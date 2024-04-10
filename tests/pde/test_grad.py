import torch
from continuiti.pde.grad import grad, Grad, div, Div


def test_grad():
    # f(x) = x_0^2 + x_1^3
    def f(x):
        return (x[:, 0] ** 2 + x[:, 1] ** 3).unsqueeze(1)

    # df(x) = [2 * x_0, 3 * x_1^2]
    def df(x):
        return torch.stack([2 * x[:, 0], 3 * x[:, 1] ** 2], dim=1)

    x = torch.rand(100, 2).requires_grad_(True)
    u = f(x)

    # Test gradient of function
    gf = grad(f)
    assert torch.norm(gf(x) - df(x)) < 1e-6

    # Test gradient operator
    du = Grad()(x, u, x)
    assert torch.norm(du - df(x)) < 1e-6


def test_div():
    # f(x) = x_0^2 + x_1^3
    def f(x):
        return (x[:, 0] ** 2 + x[:, 1] ** 3).unsqueeze(1)

    # div_f(x) = 2 * x_0 + 3 * x_1^2
    def div_f(x):
        return (2 * x[:, 0] + 3 * x[:, 1] ** 2).unsqueeze(1)

    x = torch.rand(100, 2).requires_grad_(True)
    u = f(x)

    # Test divergence of function
    df = div(f)
    assert torch.norm(df(x) - div_f(x)) < 1e-6

    # Test divergence operator
    div_u = Div()(x, u, x)
    assert torch.norm(div_u - div_f(x)) < 1e-6
