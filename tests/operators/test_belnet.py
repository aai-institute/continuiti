import torch
import pytest

from continuity.operators import BelNet
from continuity.benchmarks.sine import SineBenchmark
from continuity.data import OperatorDataset
from continuity.trainer import Trainer
from continuity.operators.losses import MSELoss

from .util import eval_shapes_correct


def test_shapes(random_shape_operator_datasets):
    assert eval_shapes_correct(BelNet, random_shape_operator_datasets)


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
        x=torch.rand((set_size, n_sensors, x_dim)),
        u=torch.rand((set_size, n_sensors, u_dim)),
        y=torch.rand((set_size, n_evals, y_dim)),
        v=torch.rand((set_size, n_evals, v_dim)),
    )

    model = BelNet(dset.shapes)

    x, u, y, v = dset[:batch_size]

    v_pred = model(x, u, y)

    assert v_pred.shape == v.shape

    y_other = torch.rand((batch_size, n_evals, y_dim))
    v_other = torch.rand((batch_size, n_evals, v_dim))

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
