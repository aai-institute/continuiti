import pytest
import torch
from typing import List
from itertools import product
from continuity.data import OperatorDataset
from continuity.benchmarks.sine import SineBenchmark
from continuity.trainer import Trainer
from continuity.operators import DeepNeuralOperator
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
def dnos(datasets) -> List[DeepNeuralOperator]:
    operators = []
    for dataset in datasets:
        operators.append(DeepNeuralOperator(dataset.shapes))
    return operators


def test_can_initialize(dnos):
    for operator in dnos:
        assert isinstance(operator, DeepNeuralOperator)


def test_output_shape_correct(dnos, datasets):
    for operator, dataset in zip(dnos, datasets):
        x, u, y, v = dataset[:9]  # batched sample
        output = operator(x, u, y)
        assert output.shape == v.shape


@pytest.mark.slow
def test_does_converge():
    # Data set
    benchmark = SineBenchmark(n_test=1, n_train=1)
    dataset = benchmark.train_dataset

    # Operator
    operator = DeepNeuralOperator(dataset.shapes)

    # Train
    Trainer(operator).fit(dataset, tol=1e-3, batch_size=1)

    # Check solution
    x, u = dataset.x, dataset.u
    assert MSELoss()(operator, x, u, x, u) < 1e-3
