from pinn_benchmark.test_cases.poisson import PoissonProblem
from pinn_benchmark.solvers.fem.dolfinx.solver import solve_dolfinx

poisson_problem = PoissonProblem(backend="dolfinx")

# Loop over the different resolutions for the DolfinX solver
with open("output/dolfinx_poisson.txt", "w") as outfile:
    for resolution in [64, 128, 256, 512, 1024]:
        print(f"Running Dolfinx with resolution of {resolution}")
        error, wall_time = solve_dolfinx(poisson_problem, resolution)
        outfile.write(f"{error}\t{wall_time}\n")
