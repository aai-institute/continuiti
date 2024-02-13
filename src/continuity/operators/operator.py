"""In Continuity, all models for operator learning are based on the `Operator` base class."""

import torch
from abc import abstractmethod
from time import time
from typing import Optional, List
from continuity.callbacks import Callback, PrintTrainingLoss
from continuity.operators.losses import Loss, MSELoss
from continuity.data import device


class Operator(torch.nn.Module):
    """Operator base class.

    An operator is a neural network model that maps functions by mapping an
    observation to the evaluations of the mapped function at given coordinates.

    This class implements default `compile` and `fit` methods.
    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Sensor positions of shape (batch_size, num_sensors, input_coordinate_dim)
            u: Input function values of shape (batch_size, num_sensors, input_channels)
            y: Coordinates where the mapped function is evaluated of shape (batch_size, y_size, output_coordinate_dim)

        Returns:
            Evaluations of the mapped function with shape (batch_size, y_size, output_channels)
        """

    def compile(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Loss] = None,
        verbose: bool = True,
    ):
        """Compile operator.

        Args:
            verbose: Print number of model parameters to stdout.
            optimizer: Torch-like optimizer.
            loss_fn: Loss function taking (x, u, y, v). Defaults to MSELoss.
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn or MSELoss()

        # Print number of model parameters
        if verbose:
            num_params = sum(p.numel() for p in self.parameters())
            print(f"Model parameters: {num_params}")

    def fit(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs: int,
        callbacks: Optional[List[Callback]] = None,
    ):
        """Fit operator to data set.

        Args:
            batch_size: Batch size.
            dataset: Data set.
            epochs: Number of epochs.
            callbacks: List of callbacks.
        """
        # Default callback
        if callbacks is None:
            callbacks = [PrintTrainingLoss()]

        # Call on_train_begin
        for callback in callbacks:
            callback.on_train_begin()

        # Train
        for epoch in range(epochs + 1):
            loss_train = 0

            start = time()
            for x, u, y, v in data_loader:
                x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)

                def closure(x=x, u=u, y=y, v=v):
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(self, x, u, y, v)
                    loss.backward(retain_graph=True)
                    return loss

                self.optimizer.step(closure)
                self.optimizer.param_groups[0]["lr"] *= 0.999

                # Compute mean loss
                loss_train += self.loss_fn(self, x, u, y, v).detach().item()

            end = time()
            seconds_per_epoch = end - start
            loss_train /= len(data_loader)

            # Callbacks
            logs = {
                "loss/train": loss_train,
                "seconds_per_epoch": seconds_per_epoch,
            }

            for callback in callbacks:
                callback(epoch, logs)

        # Call on_train_end
        for callback in callbacks:
            callback.on_train_end()

    def loss(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate loss function.

        Args:
            x: Tensor of sensor positions of shape (batch_size, num_sensors, coordinate_dim)
            u: Tensor of sensor values of shape (batch_size, num_sensors, num_channels)
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, x_size, coordinate_dim)
            v: Tensor of labels of shape (batch_size, x_size, num_channels)
        """
        return self.loss_fn(self, x, u, y, v)
