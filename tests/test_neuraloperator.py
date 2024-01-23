import torch
import matplotlib.pyplot as plt
from continuity.data.datasets import Sine
from continuity.operators import NeuralOperator
from continuity.plotting import plot, plot_evaluation

# Set random seed
torch.manual_seed(0)


def test_neuraloperator():
    # Parameters
    num_sensors = 16

    # Data set
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
    x, u, _, _ = dataset[0]  # first batch
    x0, u0 = x[0], u[0]  # first sample
    plot(x0, u0, ax=ax)
    plot_evaluation(operator, x0, u0, ax=ax)
    fig.savefig(f"test_neuraloperator.png")

    # Check solution
    assert operator.loss(x, u, x, u) < 1e-5


if __name__ == "__main__":
    test_neuraloperator()
