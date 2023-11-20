import numpy as np
from pinn_benchmark.test_cases.problem import Problem
from pinn_benchmark.test_cases.domains.unitsquare import UnitSquareDomain
from pinn_benchmark.test_cases.boundary.zero import ZeroBoundaryCondition

# Define sin and pi
sin = np.sin
pi = np.pi


class PoissonProblem(Problem):
    def __init__(self, backend: str):
        global sin
        self.periods = 4
        self.backend = backend

        # Import math functions depending on the backend
        if backend == "dolfinx":
            import ufl

            sin = ufl.sin
        elif backend == "torch":
            import torch

            sin = torch.sin
        else:
            raise NotImplementedError(f"Unknown backend: {backend}")

        # Define the source term
        def source_term(x):
            f = 1
            for i in range(self.domain.dim):
                f *= sin(self.periods * pi * x[i])
            return f

        # Define the exact solution
        def exact_solution(x):
            sine = ufl.sin if self.backend == "dolfinx" else np.sin
            u = 1
            for i in range(self.domain.dim):
                u *= sine(self.periods * pi * x[i])
            u /= 2 * self.periods**2 * pi**2
            return u

        # Initialize the parent class
        super().__init__(
            UnitSquareDomain(),
            ZeroBoundaryCondition(),
            source_term,
            exact_solution,
        )

    def pde(self, x, solution):
        if self.backend == "dolfinx":
            from ufl import inner, grad, dx

            u, v = solution
            a = inner(grad(u), grad(v)) * dx
            f = self.source_term(x)
            b = inner(f, v) * dx
            return a, b

        if self.backend == "torch":
            import deepxde as dde

            dy_xx = dde.grad.hessian(solution, x, i=0, j=0)
            dy_yy = dde.grad.hessian(solution, x, i=1, j=1)
            f = sin(self.periods * pi * x[:, 0:1]) * sin(self.periods * pi * x[:, 1:2])
            return -dy_xx - dy_yy - f

        else:
            raise NotImplementedError(f"Unknown backend: {self.backend}")
