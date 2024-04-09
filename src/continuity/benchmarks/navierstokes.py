"""
`continuity.benchmarks.navier_stokes`

Navier-Stokes benchmark.
"""

import torch
import scipy.io
import pathlib
import continuity
from typing import Optional
from continuity.benchmarks import Benchmark
from continuity.operators.losses import MSELoss
from continuity.data.dataset import OperatorDataset


class NavierStokes(Benchmark):
    r"""Navier-Stokes benchmark.

    This benchmark contains a dataset of turbulent flow samples taken from
    [neuraloperator/graph-pde](https://github.com/neuraloperator/graph-pde).

    Reference: Li, Zongyi, et al. "Neural operator: Graph kernel network for
    partial differential equations." arXiv preprint arXiv:2003.03485 (2020).

    The dataset loads the `NavierStokes_V1e-5_N1200_T20` file which contains
    of 1200 samples of 64x64 resolution with 20 time steps.

    Args:
        dir: Path to data set. Default is `data/navierstokes`
            in the root directory of the repository.
    """

    def __init__(self, dir: Optional[str] = None):
        if dir is None:
            # Get root dir relative to this file
            root_dir = pathlib.Path(continuity.__file__).parent.parent.parent
            dir = root_dir / "data" / "navierstokes"
        else:
            dir = pathlib.Path(dir)

        # Create space-time grids (x_1, x_2, t)
        ls = torch.linspace(-1, 1, 64)
        tx = torch.linspace(-1, 0, 10)
        grid_x = torch.meshgrid(ls, ls, tx, indexing="ij")
        x = torch.stack(grid_x, axis=3).reshape(1, -1, 3).repeat(1200, 1, 1)
        assert x.shape == (1200, 64 * 64 * 10, 3)

        ty = torch.linspace(0, 1, 10)
        grid_y = torch.meshgrid(ls, ls, ty, indexing="ij")
        y = torch.stack(grid_y, axis=3).reshape(1, -1, 3).repeat(1200, 1, 1)
        assert y.shape == (1200, 64 * 64 * 10, 3)

        # Load vorticity
        data = scipy.io.loadmat(dir / "NavierStokes_V1e-5_N1200_T20.mat")
        vort0 = torch.tensor(data["a"], dtype=torch.float32)
        vort = torch.tensor(data["u"], dtype=torch.float32)
        assert vort0.shape == (1200, 64, 64)
        assert vort.shape == (1200, 64, 64, 20)

        # Input is vorticity for t \in [0, 10]
        u = torch.cat(
            (vort0.reshape(-1, 64, 64, 1), vort[:, :, :, :9]),
            axis=3,
        ).reshape(1200, 64 * 64 * 10, 1)

        # Output is vorticity for t \in [10, 20]
        v = vort[:, :, :, 10:].reshape(1200, 64 * 64 * 10, 1)

        # Split train/test
        test_indices = torch.randint(1200, (200,))
        test_indices_set = set(test_indices)
        train_indices = [
            i.item() for i in torch.arange(1200) if i.item() not in test_indices_set
        ]

        train_dataset = OperatorDataset(
            x=x[train_indices],
            u=u[train_indices],
            y=y[train_indices],
            v=v[train_indices],
        )

        test_dataset = OperatorDataset(
            x=x[test_indices],
            u=u[test_indices],
            y=y[test_indices],
            v=v[test_indices],
        )

        super().__init__(train_dataset, test_dataset, [MSELoss()])
