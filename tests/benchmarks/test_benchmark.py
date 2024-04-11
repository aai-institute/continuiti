import torch
from continuiti.benchmarks import Benchmark
from continuiti.data import OperatorDataset
from continuiti.operators.losses import MSELoss


def test_can_initialize():
    n_observations = 2
    dataset = OperatorDataset(
        x=torch.rand(n_observations, 1, 1),
        u=torch.rand(n_observations, 1, 1),
        y=torch.rand(n_observations, 1, 1),
        v=torch.rand(n_observations, 1, 1),
    )

    random_benchmark = Benchmark(
        train_dataset=dataset, test_dataset=dataset, losses=[MSELoss()] * 3
    )
    assert isinstance(random_benchmark, Benchmark)
