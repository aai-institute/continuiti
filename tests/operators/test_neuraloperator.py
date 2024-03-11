import pytest
import torch
from continuity.data.sine import Sine
from continuity.operators import NeuralOperator
from continuity.trainer import Trainer
from continuity.operators.losses import MSELoss


@pytest.mark.slow
def test_neuraloperator():
    # Parameters
    num_sensors = 16

    # Data set
    dataset = Sine(num_sensors, size=1)

    # Operator
    operator = NeuralOperator(
        shapes=dataset.shapes,
        depth=1,
        kernel_width=32,
        kernel_depth=3,
    )

    # Train
    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-2)
    Trainer(operator, optimizer).fit(dataset, tol=1e-3)

    # Check solution
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 1e-3


if __name__ == "__main__":
    test_neuraloperator()
