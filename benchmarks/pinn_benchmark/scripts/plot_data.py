import matplotlib.pyplot as plt

neurons_per_layer = [4, 16, 64, 128]


def read_data(file_name):
    errors, times = [], []
    with open(file_name, "r") as infile:
        for line in infile:
            error, wall_time = map(float, line.strip().split())
            errors.append(error)
            times.append(wall_time)
    return errors, times


# Create a scatter plot of the errors vs. wall times
plt.figure()

dolfinx_errors, dolfinx_times = read_data("output/dolfinx_poisson.txt")
plt.scatter(dolfinx_times, dolfinx_errors, label="DolfinX")
plt.plot(dolfinx_times, dolfinx_errors, "-", alpha=1)

for neurons in neurons_per_layer:
    deepxde_errors, deepxde_times = read_data(f"output/deepxde_poisson_{neurons}.txt")
    plt.scatter(deepxde_times, deepxde_errors, label=f"DeepXDE {neurons}")
    plt.plot(deepxde_times, deepxde_errors, "-", alpha=0.1)

plt.xlabel("Wall time (s)")
plt.ylabel("Error")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Error vs. Wall time")
plt.savefig("output/error_vs_wall_time_poisson.png", dpi=300)
plt.show()
