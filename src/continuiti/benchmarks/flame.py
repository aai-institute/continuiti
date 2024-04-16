"""
`continuiti.benchmarks.flame`

Flame benchmark.
"""

import os
import torch
import numpy as np
import pandas as pd
import pathlib
import continuiti
from typing import List, Tuple, Optional
from continuiti.benchmarks import Benchmark
from continuiti.operators.losses import MSELoss
from continuiti.operators.shape import OperatorShapes, TensorShape
from continuiti.data.dataset import OperatorDatasetBase


class Flame(Benchmark):
    r"""Flame benchmark.

    The `Flame` benchmark contains the dataset of the
    [2023 FLAME AI Challenge](https://www.kaggle.com/competitions/2023-flame-ai-challenge)
    on super-resolution for turbulent flows.

    Args:
        flame_dir: Path to FLAME data set. Default is `data/flame` in the root directory of the repository.
        train_size: Limit size of training set. By default use the full data set.
        val_size: Limit size of validation set. By default use the full data set.
        normalize: Normalize data.
        upsample: Upsample training set.
    """

    def __init__(
        self,
        flame_dir: Optional[str] = None,
        train_size: int = None,
        val_size: int = None,
        normalize: bool = True,
        upsample: bool = False,
    ):
        if flame_dir is None:
            # Get root dir relative to this file
            root_dir = pathlib.Path(continuiti.__file__).parent.parent.parent
            flame_dir = root_dir / "data" / "flame"
        else:
            flame_dir = pathlib.Path(flame_dir)

        kwargs = {
            "flame_dir": flame_dir,
            "normalize": normalize,
            "upsample": upsample,
        }

        train_dataset = FlameDataset(split="train", size=train_size, **kwargs)
        test_dataset = FlameDataset(split="val", size=val_size, **kwargs)

        super().__init__(train_dataset, test_dataset, [MSELoss()])


class FlameDataset(OperatorDatasetBase):
    """Flame data set.

    Args:
        flame_dir: Path to data set, e.g. "data/flame/".
        split: Split. Either "train", "val" or "test".
        size: Limit size of data set. If None, use all data.
        channels: channels to load, e.g. ["rho", "ux", "uy", "uz"].
        normalize: Normalize data.
        upsample: Upsample input to 128x128 using bilinear interpolation.
    """

    all_channels = ["rho", "ux", "uy", "uz"]

    def __init__(
        self,
        flame_dir: str,
        split: str,
        size: int = None,
        channels: List[str] = all_channels,
        normalize: bool = True,
        upsample: bool = False,
    ):
        self.path = os.path.join(flame_dir, "")
        assert os.path.exists(self.path), f"Path '{self.path}' does not exist."

        self.split = split
        assert split in ["train", "val", "test"], f"Invalid split '{split}'."

        self.size = self._get_size()
        if size is not None:
            self.size = min(self.size, size)

        self.channels = channels
        assert len(channels) > 0, "Must load at least one channel."
        assert all([q in self.all_channels for q in channels]), "Invalid channel."

        self.normalize = normalize
        self.upsample = upsample

        self.upsample_layer = torch.nn.Upsample(scale_factor=8, mode="bilinear")

        size = (16, 16) if not upsample else (128, 128)
        self.shapes = OperatorShapes(
            x=TensorShape(dim=2, size=size),
            u=TensorShape(dim=len(channels), size=size),
            y=TensorShape(dim=2, size=(128, 128)),
            v=TensorShape(dim=len(channels), size=(128, 128)),
        )

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            number of samples in the data set.
        """
        return self.size

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves the input-output pair at the specified index and applies transformations.

        Parameters:
            - idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the three input tensors and the output tensor for the given index.
        """
        assert idx >= 0 and idx < self.size, f"Invalid index '{idx}'."

        # Load data
        u_i = self._load(index=idx, res="LR")
        v_i = self._load(index=idx, res="HR")

        # Normalize
        if self.normalize:

            def normalized(x):
                mean = x.mean(dim=0, keepdim=True)
                std = x.std(dim=0, keepdim=True) + 1e-3
                return (x - mean) / std

            u_i = normalized(u_i)
            v_i = normalized(v_i)

        # Positions and (optional) upsampling
        if self.upsample:
            # Upsample
            u_i = u_i.reshape(16, 16, -1)
            u_i = u_i.unsqueeze(0).swapaxes(1, -1)
            u_i = self.upsample_layer(u_i)
            u_i = u_i.swapaxes(1, -1).reshape(128 * 128, -1)

            x_i = self._create_position_grid(128)
        else:
            x_i = self._create_position_grid(16)

        y_i = self._create_position_grid(128)

        return x_i, u_i, y_i, v_i

    def _get_size(self) -> int:
        """Get size of data set."""
        csv_file = self.path + self.split + ".csv"
        data_frame = pd.read_csv(csv_file)
        return len(data_frame)

    def _load(self, index: int, res: str) -> torch.Tensor:
        """Load a flow sample from file.

        Args:
            index: Index of sample.
            res: Resolution. Either "LR" or "HR".

        Returns:
            Tensor of shape (16**2, num_channels) for LR and (128**2, num_channels) for HR.
        """
        assert res in ["LR", "HR"], f"Invalid resolution '{res}'."

        # Load file name
        csv_file = self.path + self.split + ".csv"
        data_frame = pd.read_csv(csv_file)
        data_path = self.path + f"flowfields/{res}/{self.split}/"

        xy = (16, 16) if res == "LR" else (128, 128)
        num_channels = len(self.channels)
        flow_fields = torch.zeros(*xy, num_channels)

        # Load data
        for i in range(num_channels):
            c = self.channels[i]
            filename = data_frame[f"{c}_filename"][index]
            flow_field = np.fromfile(data_path + filename, dtype="<f4").reshape(xy)
            flow_fields[:, :, i] = torch.tensor(flow_field)

        return flow_fields

    def _create_position_grid(self, size: int) -> torch.Tensor:
        """Create a flattened grid of positions in $[-1, 1]^d$."""
        ls = torch.linspace(-1, 1, size)
        mg = torch.stack(torch.meshgrid(ls, ls, indexing="ij"), axis=2)
        return mg.swapaxes(0, -1).reshape(2, size, size)
