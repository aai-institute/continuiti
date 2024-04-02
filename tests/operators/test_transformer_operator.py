import pytest

from continuity.benchmarks.sine import SineBenchmark
from continuity.trainer import Trainer
from continuity.operators.losses import MSELoss
from continuity.operators import TransformerOperator

from .util import get_shape_mismatches


def test_shapes(random_shape_operator_datasets):
    operators = [
        TransformerOperator(dataset.shapes)
        for dataset in random_shape_operator_datasets
    ]
    assert get_shape_mismatches(operators, random_shape_operator_datasets) == []


@pytest.mark.slow
def test_convergence():
    # Data set
    dataset = SineBenchmark(n_train=1).train_dataset

    # Operator
    operator = TransformerOperator(dataset.shapes)

    # Train
    Trainer(operator).fit(dataset, tol=1e-3)

    # Check solution
    x, u, y, v = dataset[:]
    assert MSELoss()(operator, x, u, y, v) < 1e-2
