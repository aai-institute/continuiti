from pinn_benchmark.test_cases.poisson import PoissonProblem
from pinn_benchmark.solvers.pinn.deepxde.solver import solve_deepxde

poisson_problem = PoissonProblem(backend="torch")

neurons_per_layer = [128]

# Loop over the different parameters for the DeepXDE solver
for neurons in neurons_per_layer:
    with open(f"output/deepxde_poisson_{neurons}.txt", "w") as outfile:
        for quadrature_points in [20]:
            print(
                f"Running DeepXDE with {neurons} neurons per layer "
                f"and {quadrature_points} quadrature points"
            )
            error, wall_time = solve_deepxde(
                poisson_problem, neurons, quadrature_points
            )
            outfile.write(f"{error}\t{wall_time}\n")
