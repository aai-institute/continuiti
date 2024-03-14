import pytest
from continuity.data.shape import TensorShape, DatasetShapes
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
    obs = shapes.num_observations

    latent_channels = 3
    x1 = x2 = shapes.x
    u1 = TensorShape(shapes.u.num, latent_channels)
    u2 = TensorShape(shapes.u.num, latent_channels)

    shapes = [
        DatasetShapes(obs, x=shapes.x, u=shapes.u, y=x1, v=u1),
        DatasetShapes(obs, x=x1, u=u1, y=x2, v=u2),
        DatasetShapes(obs, x=x2, u=u2, y=shapes.y, v=shapes.v),
    ]

    # Operator
    layers = [
        NaiveIntegralKernel(NeuralNetworkKernel(shapes[i], 32, 3)) for i in range(3)
    ]
    operator = NeuralOperator(dataset.shapes, layers)

    # Train
    Trainer(operator).fit(dataset, tol=1e-3)

    # Check solution
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 1e-3


if __name__ == "__main__":
    test_neuraloperator()
