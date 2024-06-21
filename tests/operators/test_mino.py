import pytest
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.operators.mino import MINO, AttentionKernel
from continuiti.trainer import Trainer
from .util import get_shape_mismatches


def test_shapes(random_shape_operator_datasets):
    operators = []
    for dataset in random_shape_operator_datasets:
        operators.append(AttentionKernel(dataset.shapes))
    assert get_shape_mismatches(operators, random_shape_operator_datasets) == []

    operators = []
    for dataset in random_shape_operator_datasets:
        operators.append(MINO(dataset.shapes))
    assert get_shape_mismatches(operators, random_shape_operator_datasets) == []


@pytest.mark.slow
def test_mino():
    dataset = SineBenchmark(n_train=1).train_dataset
    operator = MINO(dataset.shapes)
    logs = Trainer(operator).fit(dataset, tol=1e-2)
    logs.loss_train < 1e-2
