"""In Continuity, all models for operator learning are based on the `Operator` base class."""

import torch
from abc import abstractmethod
from time import time
from typing import Optional
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from continuity.data import device, DataSet
from continuity.operators.losses import Loss, MSELoss


class Operator(torch.nn.Module):
    """Operator base class.

    An operator is a neural network model that maps functions by mapping an
    observation to the evaluations of the mapped function at given coordinates.

    This class implements default `compile` and `fit` methods.
    """

    @abstractmethod
    def forward(self, x: Tensor, u: Tensor, y: Tensor) -> Tensor:
        """Forward pass through the operator.

        Args:
            x: Tensor of sensor positions of shape (batch_size, num_sensors, input_coordinate_dim)
            u: Tensor of sensor values of shape (batch_size, num_sensors, input_channels)
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, y_size, output_coordinate_dim)

        Returns:
            Tensor of evaluations of the mapped function of shape (batch_size, y_size, output_channels)
        """

    def compile(self, optimizer: torch.optim.Optimizer, loss_fn: Optional[Loss] = None):
        """Compile operator.

        Args:
            optimizer: Torch-like optimizer.
            loss_fn: Loss function taking (x, u, y, v). Defaults to MSELoss.
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn or MSELoss()

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
                x, u, y, v = dataset[i]

                def closure(x=x, u=u, y=y, v=v):
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(self, x, u, y, v)
                    loss.backward(retain_graph=True)
                    return loss

                self.optimizer.step(closure)
                self.optimizer.param_groups[0]["lr"] *= 0.999

                # Compute mean loss
                mean_loss += self.loss_fn(self, x, u, y, v).detach().item()

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

    def loss(self, x: Tensor, u: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Evaluate loss function.

        Args:
            x: Tensor of sensor positions of shape (batch_size, num_sensors, coordinate_dim)
            u: Tensor of sensor values of shape (batch_size, num_sensors, num_channels)
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, x_size, coordinate_dim)
            v: Tensor of labels of shape (batch_size, x_size, num_channels)
        """
        return self.loss_fn(self, x, u, y, v)
