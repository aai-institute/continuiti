"""
`continuiti.benchmarks.navier_stokes`

Navier-Stokes benchmark.
"""

import torch
import scipy.io
import pathlib
import continuiti
from typing import Optional
from continuiti.benchmarks import Benchmark
from continuiti.operators.losses import RelativeL1Error
from continuiti.data.dataset import OperatorDataset


class NavierStokes(Benchmark):
    r"""Navier-Stokes benchmark.

    This benchmark contains a dataset of turbulent flow samples taken from
    [neuraloperator/graph-pde](https://github.com/neuraloperator/graph-pde)
    that was used as illustrative example in the FNO paper:

    _Li, Zongyi, et al. "Fourier neural operator for parametric partial
    differential equations." arXiv preprint arXiv:2010.08895 (2020)_.

    The dataset loads the `NavierStokes_V1e-5_N1200_T20` file which contains
    1200 samples of Navier-Stokes flow simulations at a spatial resolution of
    64x64 and 20 time steps.

    The benchmark exports operator datasets where both input and output function
    are defined on the space-time domain (periodic in space), i.e.,
    $(x, y, t) \in [-1, 1] \times [-1, 1] \times (-1, 0]$ for the input
    function and $(x, y, t) \in [-1, 1] \times [-1, 1] \times (0, 1]$ for
    the output function.

    The input function is given by the vorticity field at the first ten time
    steps $(-0.9, -0.8, ..., 0.0)$ and the output function by the vorticity
    field at the following ten time steps $(0.1, 0.2, ..., 1.0)$.

    ![Visualization of first training sample.](/continuiti/benchmarks/img/navierstokes.png)

    The datasets have the following shapes:

    ```
        len(benchmark.train_dataset) == 1000
        len(benchmark.test_dataset) == 200

        x.shape == (3, 64, 64, 10)
        u.shape == (1, 64, 64, 10)
        y.shape == (3, 64. 64, 10)
        v.shape == (1, 64, 64, 10)
    ```

    Args:
        dir: Path to data set. Default is `data/navierstokes`
            in the root directory of the repository.
    """

    def __init__(self, dir: Optional[str] = None):
        if dir is None:
            # Get root dir relative to this file
            root_dir = pathlib.Path(continuiti.__file__).parent.parent.parent
            dir = root_dir / "data" / "navierstokes"
        else:
            dir = pathlib.Path(dir)

        # Create space-time grids (x_1, x_2, t)
        ls = torch.linspace(-1, 1, 64)
        tx = torch.linspace(-0.9, 0.0, 10)
        grid_x = torch.meshgrid(ls, ls, tx, indexing="ij")
        x = torch.stack(grid_x, axis=0).unsqueeze(0).expand(1200, -1, -1, -1, -1)
        x = x.reshape(1200, 3, 64, 64, 10)

        ty = torch.linspace(0.1, 1.0, 10)
        grid_y = torch.meshgrid(ls, ls, ty, indexing="ij")
        y = torch.stack(grid_y, axis=0).unsqueeze(0).expand(1200, -1, -1, -1, -1)
        y = y.reshape(1200, 3, 64, 64, 10)

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
        ).reshape(1200, 1, 64, 64, 10)

        # Output is vorticity for t \in [10, 20]
        v = vort[:, :, :, 10:].reshape(1200, 1, 64, 64, 10)

        # Split train/test
        train_indices = torch.arange(1000)
        test_indices = torch.arange(1000, 1200)

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

        super().__init__(train_dataset, test_dataset, [RelativeL1Error()])
