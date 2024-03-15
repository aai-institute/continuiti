import pytest
from continuity.operators.shape import TensorShape, OperatorShapes
from continuity.benchmarks.sine import SineBenchmark
from continuity.operators.integralkernel import NaiveIntegralKernel, NeuralNetworkKernel
from continuity.operators import NeuralOperator
from continuity.trainer import Trainer
from continuity.operators.losses import MSELoss


@pytest.mark.slow
def test_neuraloperator():
    # Data set
    dataset = SineBenchmark(n_train=1).train_dataset
    shapes = dataset.shapes

    latent_channels = 1
    hidden_shape = TensorShape(shapes.u.num, latent_channels)

    shapes = [
        OperatorShapes(x=shapes.x, u=shapes.u, y=shapes.x, v=hidden_shape),
        OperatorShapes(x=shapes.x, u=hidden_shape, y=shapes.x, v=hidden_shape),
        OperatorShapes(x=shapes.x, u=hidden_shape, y=shapes.y, v=shapes.v),
    ]

    # Operator
    layers = [
        NaiveIntegralKernel(NeuralNetworkKernel(shapes[i], 32, 3))
        for i in range(len(shapes))
    ]
    operator = NeuralOperator(dataset.shapes, layers)

    # Train
    Trainer(operator).fit(dataset, tol=1e-3)

    # Check solution
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 1e-3


if __name__ == "__main__":
    test_neuraloperator()
