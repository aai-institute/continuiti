import pytest
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.trainer import Trainer
from continuiti.operators import FourierNeuralOperator
from continuiti.operators.losses import MSELoss


@pytest.mark.slow
def test_fno():
    # Data set
    benchmark = SineBenchmark(n_train=1)
    dataset = benchmark.train_dataset

    # Operator
    operator = FourierNeuralOperator(dataset.shapes, depth=3, width=3)

    # Train
    Trainer(operator, device="cpu").fit(dataset, tol=1e-12, epochs=10_000)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-12
