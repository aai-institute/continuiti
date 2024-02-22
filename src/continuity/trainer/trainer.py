import torch
from time import time
from typing import Optional, List
from continuity.operators import Operator
from continuity.operators.losses import Loss, MSELoss
from continuity.trainer.callbacks import Callback, PrintTrainingLoss


class Trainer:
    """Trainer.

    Implements a default training loop for operator learning.

    Example:
        ```python
        from continuity.trainer import Trainer
        from continuity.operators.losses import MSELoss
        ...
        optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3)
        loss_fn = MSELoss()
        trainer = Trainer(operator, optimizer, loss_fn, device="cuda:0")
        trainer.fit(data_loader, epochs=100)
        ```

    Args:
        operator: Operator to be trained.
        optimizer: Torch-like optimizer. Default is Adam.
        criterion: Loss function taking (op, x, u, y, v). Default is MSELoss.
        device: Device to train on. Default is CPU.
    """

    def __init__(
        self,
        operator: Operator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Loss] = None,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ):
        self.operator = operator
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(operator.parameters(), lr=1e-3)
        )
        self.loss_fn = loss_fn if loss_fn is not None else MSELoss()
        self.device = device if device is not None else torch.device("cpu")
        self.verbose = verbose

    def fit(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        callbacks: Optional[List[Callback]] = None,
    ):
        """Fit operator to data set.

        Args:
            dataset: Data set.
            epochs: Number of epochs.
            callbacks: List of callbacks.
        """
        # Default callback
        if callbacks is None:
            if self.verbose:
                callbacks = [PrintTrainingLoss()]
            else:
                callbacks = []

        # Print number of model parameters
        if self.verbose:
            num_params = sum(p.numel() for p in self.operator.parameters())
            print(f"Model parameters: {num_params}")

        # Move operator to device
        self.operator.to(self.device)

        # Call on_train_begin
        for callback in callbacks:
            callback.on_train_begin()

        # Train
        self.operator.train()
        for epoch in range(epochs):
            loss_train = 0

            start = time()
            for x, u, y, v in data_loader:
                x, u = x.to(self.device), u.to(self.device)
                y, v = y.to(self.device), v.to(self.device)

                def closure(x=x, u=u, y=y, v=v):
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(self.operator, x, u, y, v)
                    loss.backward(retain_graph=True)
                    return loss

                self.optimizer.step(closure)

                # Compute mean loss
                loss_train += self.loss_fn(self.operator, x, u, y, v).detach().item()

            end = time()
            seconds_per_epoch = end - start
            loss_train /= len(data_loader)

            # Callbacks
            logs = {
                "loss/train": loss_train,
                "seconds_per_epoch": seconds_per_epoch,
            }

            for callback in callbacks:
                callback(epoch + 1, logs)

        # Call on_train_end
        for callback in callbacks:
            callback.on_train_end()

        # Move operator back to CPU
        self.operator.to("cpu")
