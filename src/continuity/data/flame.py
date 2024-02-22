"""
`continuity.data.flame`

Flame data set.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import List

from continuity.data import OperatorDataset


class FlameDataLoader:
    """Data loader for turbulent flow samples from flame dataset.

    Args:
        path: Path to data set.
    """

    all_channels = ["ux", "uy", "uz", "rho"]

    def __init__(self, path: str = "flame/"):
        self.path = os.path.join(path, "")
        assert os.path.exists(self.path), f"Path '{self.path}' does not exist."

    def size(self, split: str = "train") -> int:
        """Get size of data set.

        Args:
            split: Split. Either "train", "val" or "test".

        Returns:
            Size of data set.
        """
        assert split in ["train", "val", "test"], f"Invalid split '{split}'."
        csv_file = self.path + split + ".csv"
        data_frame = pd.read_csv(csv_file)
        return len(data_frame)

    def load(
        self,
        index: int = 0,
        res: str = "LR",
        split: str = "train",
        channels: List[str] = all_channels,
    ) -> torch.Tensor:
        """Load a flow sample from file.

        Args:
            index: Index of sample.
            res: Resolution. Either "LR" or "HR".
            split: Split. Either "train", "val" or "test".
            channels: channels to load, e.g. ["ux", "uy", "uz", "rho"].

        Returns:
            Tensor of shape (16**2, num_channels) for LR and (128**2, num_channels) for HR.
        """
        assert res in ["LR", "HR"], f"Invalid resolution '{res}'."
        assert split in ["train", "val", "test"], f"Invalid split '{split}'."
        assert len(channels) > 0, "Must load at least one channel."
        assert all([q in self.all_channels for q in channels]), "Invalid channel."
        assert index >= 0 and index < self.size(split), f"Invalid index '{index}'."

        # Load file name
        csv_file = self.path + split + ".csv"
        data_frame = pd.read_csv(csv_file)
        data_path = self.path + f"flowfields/{res}/{split}/"

        xy = 16 * 16 if res == "LR" else 128 * 128
        num_channels = len(channels)
        flow_fields = torch.zeros(xy, num_channels)

        # Load data
        for i in range(num_channels):
            c = channels[i]
            filename = data_frame[f"{c}_filename"][index]
            flow_field = np.fromfile(data_path + filename, dtype="<f4")
            flow_fields[:, i] = torch.tensor(flow_field)

        return flow_fields


def create_position_grid(size):
    """Create a grid of positions in $[-1, 1]^d$.

    Args:
        size: Size of grid in each dimension.

    Returns:
        Tensor of shape (size, size, 2) with positions in $[-1, 1]^2$.
    """
    ls = torch.linspace(-1, 1, size)
    return torch.stack(torch.meshgrid(ls, ls, indexing="ij"), axis=2).reshape(-1, 2)


class Flame(OperatorDataset):
    """Flame data set.

    Args:
        data_loader: Data loader.
        size: Limit size of data set. If None, use all data.
        split: Split. Either "train", "val" or "test".
        channels: channels to load, e.g. ["ux", "uy", "uz", "rho"].
        normalize: Normalize data.
        batch_size: Batch size.
    """

    def __init__(
        self,
        data_loader: FlameDataLoader,
        size: int = None,
        split: str = "train",
        channels: List[str] = FlameDataLoader.all_channels,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        self.data_loader = data_loader
        self.split = split
        self.channels = channels
        self.size = self.data_loader.size(split)
        if size is not None:
            self.size = min(self.size, size)
        self.batch_size = batch_size

        x = []
        u = []
        y = []
        v = []

        for index in range(self.size):
            # Load data
            u_i = self.data_loader.load(
                index=index, res="LR", split=self.split, channels=self.channels
            )
            v_i = self.data_loader.load(
                index=index, res="HR", split=self.split, channels=self.channels
            )

            # Normalize
            if normalize:
                mean, std = u_i.mean(), u_i.std()
                u_i = (u_i - mean) / std
                v_i = (v_i - mean) / std

            # Positions
            x_i = create_position_grid(16)
            y_i = create_position_grid(128)

            # Add to list
            x.append(x_i)
            u.append(u_i)
            y.append(y_i)
            v.append(v_i)

        # Stack
        x = torch.stack(x)
        u = torch.stack(u)
        y = torch.stack(y)
        v = torch.stack(v)

        super().__init__(x, u, y, v)
