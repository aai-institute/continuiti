import pytest
import torch
from continuiti.operators import DeepONet
from continuiti.networks import DeepResidualNetwork
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.trainer import Trainer


def train():
    dataset = SineBenchmark(n_train=8).train_dataset
    operator = DeepONet(dataset.shapes, trunk_depth=8)

    Trainer(operator).fit(dataset, tol=1e-2)

    # Make sure we can use operator output on cpu again
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    v_pred = operator(x, u, y)
    assert ((v_pred - v.to("cpu")) ** 2).mean() < 1e-2


@pytest.mark.slow
def test_trainer_with_operator():
    train()


@pytest.mark.slow
def test_trainer_with_torch_model():
    def f(x):
        return torch.sin(2 * torch.pi * x)

    x_train = torch.rand(128, 1)
    x_test = torch.rand(32, 1)

    y_train = f(x_train)
    y_test = f(x_test)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    # Create a model
    model = DeepResidualNetwork(
        input_size=1,
        output_size=1,
        width=32,
        depth=8,
    )

    # Define loss function (in continuiti style)
    mse = torch.nn.MSELoss()

    def loss_fn(op, x, y):
        y_pred = op(x)
        return mse(y_pred, y)

    # Train the model
    trainer = Trainer(model, loss_fn=loss_fn)
    logs = trainer.fit(
        train_dataset,
        tol=1e-2,
        test_dataset=test_dataset,
    )

    # Test the model
    assert logs.loss_test < 1e-2


# Use ./run_parallel.sh to run test with CUDA
if __name__ == "__main__":
    test_trainer_with_torch_model()
