import pytest
import torch
from itertools import product
from typing import List
from continuiti.data import OperatorDataset


@pytest.fixture(scope="session")
def random_shape_operator_datasets() -> List[OperatorDataset]:
    """Pytest fixture returning 16 combinations of differently shaped domains, input function observations, co-domains,
    and output function observations.

    The dimensionality of each vector-space (domain, input function observation, co-domain, and output function
    observation) is one or two. All combinations (n=2^4) of different dimensionality are probed with this list of
    Operator-Datasets. The number of sensors, evaluations, and observations is constant (prime numbers) across these
    different sets of data.

    Returns:
        List containing permutations of 1D and 2D vectors (x, u, y, v) in OperatorDataset.
    """
    n_sensors = 5
    n_evaluations = 7
    n_observations = 11
    x_dims = (1, 2)
    u_dims = (1, 2)
    y_dims = (1, 2)
    v_dims = (1, 2)

    datasets = []

    for x_dim, u_dim, y_dim, v_dim in product(x_dims, u_dims, y_dims, v_dims):
        x_samples = torch.rand(n_observations, x_dim, n_sensors)
        u_samples = torch.rand(n_observations, u_dim, n_sensors)
        y_samples = torch.rand(n_observations, y_dim, n_evaluations)
        v_samples = torch.rand(n_observations, v_dim, n_evaluations)
        datasets.append(
            OperatorDataset(x=x_samples, u=u_samples, y=y_samples, v=v_samples)
        )

    # 2D grids
    for x_dim, u_dim, y_dim, v_dim in product(x_dims, u_dims, y_dims, v_dims):
        x_samples = torch.rand(n_observations, x_dim, n_sensors, n_sensors + 1)
        u_samples = torch.rand(n_observations, u_dim, n_sensors, n_sensors + 1)
        y_samples = torch.rand(n_observations, y_dim, n_evaluations, n_evaluations + 1)
        v_samples = torch.rand(n_observations, v_dim, n_evaluations, n_evaluations + 1)
        datasets.append(
            OperatorDataset(x=x_samples, u=u_samples, y=y_samples, v=v_samples)
        )

    return datasets
