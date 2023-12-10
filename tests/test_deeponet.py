import torch
import matplotlib.pyplot as plt
from continuity.data.datasets import Sine
from continuity.operators import DeepONet
from continuity.plotting import plot_observation, plot_evaluation

# Set random seed
torch.manual_seed(0)


def test_deeponet():
    # Parameters
    num_sensors = 16

    # Observation
    dataset = Sine(num_sensors, size=1)

    # Operator
    operator = DeepONet(
        num_sensors,
        dataset.coordinate_dim,
        dataset.num_channels,
        branch_width=32,
        branch_depth=1,
        trunk_width=32,
        trunk_depth=1,
        basis_functions=4,
    )

    # Train self-supervised
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()

    operator.compile(optimizer, criterion)
    operator.fit(dataset, epochs=1000)

    # Plotting
    fig, ax = plt.subplots(1, 1)
    observation = dataset.get_observation(0)
    plot_observation(observation, ax=ax)
    plot_evaluation(operator, observation, ax=ax)
    fig.savefig(f"test_deeponet.png")

    # Check solution
    x, u = observation.to_tensors()
    u_predicted = operator(x, u, x).reshape(u.shape)
    assert criterion(u_predicted, u) < 1e-5


if __name__ == "__main__":
    test_deeponet()
