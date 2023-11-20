import numpy as np
import ufl
from time import time
from dolfinx import fem, mesh
from mpi4py import MPI
from ufl import dx
from pinn_benchmark.test_cases.problem import Problem


def solve_dolfinx(problem: Problem, resolution=16):
    dim = problem.domain.dim
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(
        comm=comm,
        points=problem.domain.vertices,
        n=tuple([resolution] * dim),
        cell_type=mesh.CellType.triangle,
    )
    V = fem.FunctionSpace(msh, ("Lagrange", 1))

    bc = problem.boundary_conditions
    facets = mesh.locate_entities_boundary(
        msh, dim=dim - 1, marker=lambda x: bc.boundary(x)
    )
    dofs = fem.locate_dofs_topological(V=V, entity_dim=dim - 1, entities=facets)
    bc = fem.dirichletbc(value=np.array(0.0), dofs=dofs, V=V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    a, b = problem.pde(x, [u, v])

    lin_prob = fem.petsc.LinearProblem(
        a, b, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )

    elapsed = -time()
    uh = lin_prob.solve()
    elapsed += time()

    u_exact = problem.exact_solution(x)
    m = fem.form((u_exact - uh) ** 2 * dx)
    error = comm.allreduce(fem.assemble_scalar(m), op=MPI.SUM) ** 0.5

    print(f"L2-Error: {error:.6e}")
    print(f"Elapsed time: {elapsed:.3e} seconds")

    return error, elapsed
