import torch
import pytest

from continuiti.operators import DeepONet
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.data import OperatorDataset
from continuiti.trainer import Trainer
from continuiti.operators.losses import MSELoss
from .util import get_shape_mismatches


def test_shapes(random_shape_operator_datasets):
    operators = [DeepONet(dataset.shapes) for dataset in random_shape_operator_datasets]
    assert get_shape_mismatches(operators, random_shape_operator_datasets) == []


def test_output_shape():
    x_dim = 2
    u_dim = 3
    y_dim = 5
    v_dim = 7
    n_sensors = 11
    n_evals = 13
    batch_size = 17
    set_size = 19

    dset = OperatorDataset(
        x=torch.rand((set_size, x_dim, n_sensors)),
        u=torch.rand((set_size, u_dim, n_sensors)),
        y=torch.rand((set_size, y_dim, n_evals)),
        v=torch.rand((set_size, v_dim, n_evals)),
    )

    model = DeepONet(dset.shapes)

    x, u, y, v = dset[:batch_size]

    v_pred = model(x, u, y)

    assert v_pred.shape == v.shape

    y_other = torch.rand((batch_size, y_dim, n_evals * 5))
    v_other = torch.rand((batch_size, v_dim, n_evals * 5))

    v_other_pred = model(x, u, y_other)

    assert v_other_pred.shape == v_other.shape


@pytest.mark.slow
def test_deeponet():
    # Data set
    dataset = SineBenchmark(n_train=1).train_dataset

    # Operator
    operator = DeepONet(dataset.shapes)

    # Train
    Trainer(operator).fit(dataset, tol=1e-2)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-2


@pytest.mark.slow
def test_custom_branch_network():
    # Data set
    n_sensors = 32
    dataset = SineBenchmark(n_train=1, n_sensors=n_sensors).train_dataset

    # CNN as branch network
    basis_functions = 8
    branch_network = torch.nn.Sequential(
        torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(16, 1, kernel_size=3, padding=1),
        torch.nn.Flatten(),
        torch.nn.Linear(n_sensors, basis_functions),
    )

    # Operator
    operator = DeepONet(
        dataset.shapes,
        branch_network=branch_network,
        basis_functions=basis_functions,
    )

    # Train
    Trainer(operator).fit(dataset, tol=1e-3)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-3


@pytest.mark.slow
def test_custom_trunk_network():
    # Data set
    n_sensors = 32
    dataset = SineBenchmark(n_train=1, n_sensors=n_sensors).train_dataset

    # MLP as trunk network
    basis_functions = 32
    trunk_network = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.LayerNorm(32),
        torch.nn.Sigmoid(),
        torch.nn.Linear(32, 32),
        torch.nn.BatchNorm1d(32),
        torch.nn.GELU(),
        torch.nn.Linear(32, basis_functions),
    )

    # Operator
    operator = DeepONet(
        dataset.shapes,
        trunk_network=trunk_network,
        basis_functions=basis_functions,
    )

    # Train
    Trainer(operator).fit(dataset, tol=1e-3)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-3


@pytest.mark.slow
def test_custom_branch_and_trunk_network():
    # Data set
    n_sensors = 32
    dataset = SineBenchmark(n_train=1, n_sensors=n_sensors).train_dataset

    # CNN as branch network
    basis_functions = 32
    branch_network = torch.nn.Sequential(
        torch.nn.Conv1d(1, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv1d(16, 1, kernel_size=3, padding=1),
        torch.nn.Flatten(),
        torch.nn.Linear(n_sensors, basis_functions),
    )

    # Custom MLP as trunk network
    trunk_network = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.LayerNorm(32),
        torch.nn.Sigmoid(),
        torch.nn.Linear(32, 32),
        torch.nn.BatchNorm1d(32),
        torch.nn.GELU(),
        torch.nn.Linear(32, basis_functions),
    )

    # Operator
    operator = DeepONet(
        dataset.shapes,
        branch_network=branch_network,
        trunk_network=trunk_network,
    )

    # Train
    Trainer(operator).fit(dataset, tol=1e-3)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-3

    # Operator
    operator = DeepONet(
        dataset.shapes,
        trunk_network=trunk_network,
        basis_functions=basis_functions,
    )

    # Train
    Trainer(operator).fit(dataset, tol=1e-3)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-3
