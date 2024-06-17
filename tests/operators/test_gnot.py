import pytest
import torch

from continuiti.operators import GNOT
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.trainer import Trainer
from continuiti.operators.losses import MSELoss

from .util import get_shape_mismatches


@pytest.fixture(scope="class")
def random_dataset():
    benchmark = SineBenchmark(n_train=1)
    return benchmark.train_dataset


class TestGNOT:
    def test_can_initialize(self, random_dataset):
        operator = GNOT(random_dataset.shapes)
        assert isinstance(operator, GNOT)

    def test_can_forward(self, random_dataset):
        operator = GNOT(random_dataset.shapes)

        x, u, y, _ = next(iter(random_dataset))
        x, u, y = x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0)

        out = operator(x, u, y)

        assert isinstance(out, torch.Tensor)

    def test_gradient_flow(self, random_dataset):
        operator = GNOT(random_dataset.shapes)

        x, u, y, _ = next(iter(random_dataset))
        x, u, y = x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0)

        x.requires_grad = True
        u.requires_grad = True
        y.requires_grad = True

        out = operator(x, u, y)

        out.sum().backward()

        assert x.grad is not None, "Gradients not flowing to x"
        assert u.grad is not None, "Gradients not flowing to u"
        assert y.grad is not None, "Gradients not flowing to y"

    def test_shapes(self, random_shape_operator_datasets):
        operators = [GNOT(dataset.shapes) for dataset in random_shape_operator_datasets]
        assert get_shape_mismatches(operators, random_shape_operator_datasets) == []

    @pytest.mark.slow
    def test_can_overfit(self, random_dataset):
        operator = GNOT(random_dataset.shapes)

        # Train
        Trainer(operator).fit(random_dataset, tol=5e-3)

        # Check solution
        x, u, y, v = (
            random_dataset.x,
            random_dataset.u,
            random_dataset.y,
            random_dataset.v,
        )
        assert MSELoss()(operator, x, u, y, v) < 1e-2
