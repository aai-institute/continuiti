import pytest
from continuity.benchmarks.sine import SineBenchmark
from continuity.trainer import Trainer
from continuity.operators import FourierNeuralOperator
from continuity.operators.losses import MSELoss


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
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 1e-12


if __name__ == "__main__":
    test_fno()
