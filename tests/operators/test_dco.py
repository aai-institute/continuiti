import pytest
from typing import List

from continuiti.operators import DeepCatOperator
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.trainer import Trainer
from continuiti.operators.losses import MSELoss
from continuiti.networks import DeepResidualNetwork

from .util import get_shape_mismatches


@pytest.fixture(scope="module")
def dcos(random_shape_operator_datasets) -> List[DeepCatOperator]:
    return [
        DeepCatOperator(dataset.shapes) for dataset in random_shape_operator_datasets
    ]


class TestDeepCatOperator:
    def test_can_initialize(self, random_operator_dataset):
        operator = DeepCatOperator(random_operator_dataset.shapes)

        assert isinstance(operator, DeepCatOperator)

    def test_can_initialize_default_networks(self, random_operator_dataset):
        operator = DeepCatOperator(shapes=random_operator_dataset.shapes)

        assert isinstance(operator.branch_net, DeepResidualNetwork)
        assert isinstance(operator.trunk_net, DeepResidualNetwork)
        assert isinstance(operator.cat_net, DeepResidualNetwork)

    def test_forward_shapes_correct(self, dcos, random_shape_operator_datasets):
        assert get_shape_mismatches(dcos, random_shape_operator_datasets) == []

    @pytest.mark.slow
    def test_can_overfit(self):
        # Data set
        dataset = SineBenchmark(n_train=1).train_dataset

        # Operator
        operator = DeepCatOperator(dataset.shapes)

        # Train
        Trainer(operator).fit(dataset, tol=1e-2)

        # Check solution
        x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
        assert MSELoss()(operator, x, u, y, v) < 1e-2
