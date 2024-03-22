import pytest
from continuity.benchmarks.sine import SineBenchmark
from continuity.trainer import Trainer
from continuity.operators import ConvolutionalNeuralNetwork
from continuity.operators.losses import MSELoss


@pytest.mark.slow
def test_cnn():
    # Data set
    benchmark = SineBenchmark(n_train=1)
    dataset = benchmark.train_dataset

    # Operator
    operator = ConvolutionalNeuralNetwork(dataset.shapes)

    # Train
    Trainer(operator).fit(dataset, tol=1e-3)

    # Check solution
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 1e-3


if __name__ == "__main__":
    test_cnn()
