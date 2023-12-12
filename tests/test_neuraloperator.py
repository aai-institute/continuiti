import torch
import matplotlib.pyplot as plt
from continuity.data.datasets import Sine
from continuity.operators import NeuralOperator
from continuity.plotting import plot_observation, plot_evaluation

# Set random seed
torch.manual_seed(0)


def test_neuraloperator():
    # Parameters
    num_sensors = 16

    # Observation
    dataset = Sine(num_sensors, size=1)

    # Operator
    operator = NeuralOperator(
        coordinate_dim=dataset.coordinate_dim,
        num_channels=dataset.num_channels,
        depth=1,
        kernel_width=32,
        kernel_depth=3,
    )

    # Train self-supervised
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-2)
    operator.compile(optimizer)
    operator.fit(dataset, epochs=400)

    # Plotting
    fig, ax = plt.subplots(1, 1)
    observation = dataset.get_observation(0)
    plot_observation(observation, ax=ax)
    plot_evaluation(operator, observation, ax=ax)
    fig.savefig(f"test_neuraloperator.png")

    # Check solution
    x, u = observation.to_tensors()
    assert operator.loss(x, u, x, u) < 1e-5


if __name__ == "__main__":
    test_neuraloperator()
