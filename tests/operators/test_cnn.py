import pytest
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.trainer import Trainer
from continuiti.operators import ConvolutionalNeuralNetwork
from continuiti.operators.losses import MSELoss


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
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-3
