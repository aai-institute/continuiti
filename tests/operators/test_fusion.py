import pytest
import torch
from typing import List
from itertools import product
from continuity.operators import FusionOperator
from continuity.data import OperatorDataset
from continuity.data.sine import Sine
from continuity.trainer import Trainer
from continuity.operators.losses import MSELoss


@pytest.fixture(scope="module")
def datasets() -> List[OperatorDataset]:
    n_sensors = 3
    n_evaluations = 5
    n_observations = 7
    x_dims = (1, 2)
    u_dims = (1, 2)
    y_dims = (1, 2)
    v_dims = (1, 2)

    datasets = []

    for x_dim, u_dim, y_dim, v_dim in product(x_dims, u_dims, y_dims, v_dims):
        x_samples = torch.rand(n_observations, n_sensors, x_dim)
        u_samples = torch.rand(n_observations, n_sensors, u_dim)
        y_samples = torch.rand(n_observations, n_evaluations, y_dim)
        v_samples = torch.rand(n_observations, n_evaluations, v_dim)
        datasets.append(
            OperatorDataset(x=x_samples, u=u_samples, y=y_samples, v=v_samples)
        )

    return datasets


@pytest.fixture(scope="module")
def fusion_operators(datasets) -> List[FusionOperator]:
    operators = []
    for dataset in datasets:
        operators.append(FusionOperator(dataset.shapes))
    return operators


def test_can_initialize(fusion_operators):
    for operator in fusion_operators:
        assert isinstance(operator, FusionOperator)


def test_output_shape_correct(fusion_operators, datasets):
    for operator, dataset in zip(fusion_operators, datasets):
        x, u, y, v = dataset[:5]  # batched sample
        output = operator(x, u, y)
        assert output.shape == v.shape


def test_does_converge():
    # Parameters
    num_sensors = 16

    # Data set
    dataset = Sine(num_sensors, size=1)

    # Operator
    operator = FusionOperator(dataset.shapes)

    # Train
    Trainer(operator).fit(dataset, tol=1e-3, batch_size=1)

    # Check solution
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 1e-3
