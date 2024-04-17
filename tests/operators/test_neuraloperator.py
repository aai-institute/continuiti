import pytest
from continuiti.operators.shape import TensorShape, OperatorShapes
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.operators.integralkernel import NaiveIntegralKernel, NeuralNetworkKernel
from continuiti.operators import NeuralOperator
from continuiti.trainer import Trainer
from .util import get_shape_mismatches


def test_shapes(random_shape_operator_datasets):
    latent_channels = 3
    operators = []
    for dataset in random_shape_operator_datasets:
        shapes = dataset.shapes
        hidden_shape = TensorShape(dim=latent_channels, size=shapes.u.size)

        shapes = [
            OperatorShapes(x=shapes.x, u=hidden_shape, y=shapes.x, v=hidden_shape),
            OperatorShapes(x=shapes.x, u=hidden_shape, y=shapes.x, v=hidden_shape),
            OperatorShapes(
                x=shapes.x,
                u=hidden_shape,
                y=shapes.y,
                v=TensorShape(dim=latent_channels, size=shapes.v.size),
            ),
        ]
        layers = [
            NaiveIntegralKernel(NeuralNetworkKernel(shapes[i], 32, 3))
            for i in range(len(shapes))
        ]
        operators.append(NeuralOperator(dataset.shapes, layers))

    assert get_shape_mismatches(operators, random_shape_operator_datasets) == []


@pytest.mark.slow
def test_neuraloperator():
    # Data set
    dataset = SineBenchmark(n_train=1).train_dataset
    shapes = dataset.shapes

    # Operator
    width, depth = 7, 9
    latent_shape = OperatorShapes(
        x=shapes.x,
        u=TensorShape(dim=width, size=shapes.u.size),
        y=shapes.y,
        v=TensorShape(dim=width, size=shapes.u.size),
    )
    layers = [
        NaiveIntegralKernel(NeuralNetworkKernel(latent_shape, 32, 3))
        for _ in range(depth)
    ]
    operator = NeuralOperator(dataset.shapes, layers)

    # Train
    logs = Trainer(operator).fit(dataset, tol=1e-2)

    # Check solution
    logs.loss_train < 1e-2
