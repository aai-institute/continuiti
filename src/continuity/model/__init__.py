"""In Continuity, all models for operator learning are based on the `Operator` base class."""

import torch
from abc import abstractmethod
from time import time
from typing import Optional
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import _Loss as Loss
from continuity.data import device, DataSet


class Operator(torch.nn.Module):
    """Operator base class.

    An operator is a neural network model that maps functions by mapping an
    observation to the evaluations of the mapped function at given coordinates.

    This class implements default `compile` and `fit` methods.
    """

    @abstractmethod
    def forward(self, xu: Tensor, y: Tensor) -> Tensor:
        """Forward pass through the operator.

        Args:
            xu: Tensor of observations of shape (batch_size, num_sensors, coordinate_dim + num_channels)
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, x_size, coordinate_dim)

        Returns:
            Tensor of evaluations of the mapped function of shape (batch_size, x_size, num_channels)
        """

    def compile(self, optimizer: torch.optim.Optimizer, criterion: Loss):
        """Compile operator.

        Args:
            optimizer: Torch-like optimizer.
            criterion: Torch-like loss function taking prediction and label.
        """
        self.optimizer = optimizer
        self.criterion = criterion

        # Move to device
        self.to(device)

        # Print number of model parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {num_params}")

    def fit(
        self, dataset: DataSet, epochs: int, writer: Optional[SummaryWriter] = None
    ):
        """Fit operator to data set.

        Args:
            dataset: Data set.
            epochs: Number of epochs.
            writer: Tensorboard-like writer for loss visualization.
        """
        for epoch in range(epochs + 1):
            mean_loss = 0

            start = time()
            for i in range(len(dataset)):
                u, x, v = dataset[i]

                def closure(u=u, v=v, x=x):
                    self.optimizer.zero_grad()
                    loss = self.criterion(self(u, x), v)
                    loss.backward()
                    return loss

                self.optimizer.step(closure)
                self.optimizer.param_groups[0]["lr"] *= 0.999
                mean_loss += self.criterion(self(u, x), v).item()
            end = time()
            mean_loss /= len(dataset)

            if writer is not None:
                writer.add_scalar("Loss/train", mean_loss, epoch)

            iter_per_second = len(dataset) / (end - start)
            print(
                f"\rEpoch {epoch}:  loss = {mean_loss:.4e}  "
                f"({iter_per_second:.2f} it/s)",
                end="",
            )
        print("")
