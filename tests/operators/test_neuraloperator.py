import pytest
import torch
import matplotlib.pyplot as plt
from continuity.data import Sine
from continuity.operators import NeuralOperator
from continuity.plotting import plot, plot_evaluation
from torch.utils.data import DataLoader

# Set random seed
torch.manual_seed(0)


@pytest.mark.slow
def test_neuraloperator():
    # Parameters
    num_sensors = 16

    # Data set
    dataset = Sine(num_sensors, size=1)
    data_loader = DataLoader(dataset, batch_size=32)

    # Operator
    operator = NeuralOperator(
        shapes=dataset.shapes,
        depth=1,
        kernel_width=32,
        kernel_depth=3,
    )

    # Train self-supervised
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-2)
    operator.compile(optimizer)
    operator.fit(data_loader, epochs=400)

    # Plotting
    fig, ax = plt.subplots(1, 1)
    x, u, _, _ = dataset[0]
    plot(x, u, ax=ax)
    plot_evaluation(operator, x, u, ax=ax)
    fig.savefig(f"test_neuraloperator.png")

    # Check solution
    x = x.unsqueeze(0)
    u = u.unsqueeze(0)
    assert operator.loss(x, u, x, u) < 1e-5


if __name__ == "__main__":
    test_neuraloperator()
