import torch
import pytest

from continuiti.operators import BelNet
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.data import OperatorDataset
from continuiti.trainer import Trainer
from continuiti.operators.losses import MSELoss

from .util import get_shape_mismatches


def test_shapes(random_shape_operator_datasets):
    operators = [BelNet(dataset.shapes) for dataset in random_shape_operator_datasets]
    assert get_shape_mismatches(operators, random_shape_operator_datasets) == []


def test_belnet_shape():
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

    model = BelNet(dset.shapes)

    x, u, y, v = dset[:batch_size]

    v_pred = model(x, u, y)

    assert v_pred.shape == v.shape

    y_other = torch.rand((batch_size, y_dim, n_evals))
    v_other = torch.rand((batch_size, v_dim, n_evals))

    v_other_pred = model(x, u, y_other)

    assert v_other_pred.shape == v_other.shape


@pytest.mark.slow
def test_belnet():
    # Data set
    dataset = SineBenchmark(n_train=1).train_dataset

    # Operator
    operator = BelNet(dataset.shapes)

    # Train
    Trainer(operator).fit(dataset, tol=1e-2)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-2
