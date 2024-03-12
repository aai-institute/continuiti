"""
`continuity.trainer.trainer`
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import time
from typing import Optional, List
from continuity.data import OperatorDataset, dataset_loss
from continuity.operators import Operator
from continuity.operators.losses import Loss, MSELoss
from continuity.trainer.device import get_device
from .callbacks import Callback, PrintTrainingLoss
from .criterion import Criterion, TrainingLossCriterion
from .logs import Logs


class Trainer:
    """Trainer implements a default training loop for operator learning.

    Example:
        ```python
        from continuity.trainer import Trainer
        from continuity.operators.losses import MSELoss
        ...
        optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3)
        loss_fn = MSELoss()
        trainer = Trainer(operator, optimizer, loss_fn, device="cuda:0")
        trainer.fit(dataset, tol=1e-3, epochs=1000)
        ```

    Args:
        operator: Operator to be trained.
        lr: Learning rate. Ignored if optimizer is not None. Default is 1e-3.
        optimizer: Torch-like optimizer. Default is Adam with learning rate `lr`.
        loss_fn: Loss function taking (op, x, u, y, v). Default is MSELoss.
        device: Device to train on. Default is CPU.
        verbose: Print model parameters and use PrintTrainingLoss callback by default. Default is True.
    """

    device = get_device()

    def __init__(
        self,
        operator: Operator,
        lr: float = 1e-3,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Loss] = None,
        device: torch.device = device,
        verbose: Optional[bool] = None,
    ):
        self.operator = operator
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(operator.parameters(), lr=lr)
        )
        self.loss_fn = loss_fn if loss_fn is not None else MSELoss()
        self.device = device
        self.rank = device.index or 0 if device != "cpu" else 0
        self.verbose = verbose if verbose is not None else self.rank == 0

    def fit(
        self,
        dataset: OperatorDataset,
        tol: float = 1e-5,
        epochs: int = 1000,
        callbacks: Optional[List[Callback]] = None,
        criterion: Optional[Criterion] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        val_dataset: Optional[OperatorDataset] = None,
    ):
        """Fit operator to data set.

        Args:
            dataset: Data set.
            tol: Tolerance for stopping criterion. Ignored if criterion is not None.
            epochs: Maximum number of epochs.
            callbacks: List of callbacks. Defaults to [PrintTrainingLoss] if verbose.
            criterion: Stopping criterion. Defaults to TrainingLossCriteria(tol).
            batch_size: Batch size.
            shuffle: Shuffle data set.
            val_dataset: Validation data set.
        """
        # Default callback
        if callbacks is None:
            if self.verbose:
                callbacks = [PrintTrainingLoss()]
            else:
                callbacks = []

        # Default criterion
        if criterion is None:
            criterion = TrainingLossCriterion(tol)

        # Print number of model parameters
        if self.verbose:
            num_params = sum(p.numel() for p in self.operator.parameters())
            print(f"Model parameters: {num_params}")

        # Move operator to device
        operator = self.operator.to(self.device)

        # Use DistributedDataParallel if available
        sampler = None
        if dist.is_available() and dist.is_initialized():
            operator = DDP(
                operator, device_ids=[self.device], output_device=self.device
            )
            sampler = DistributedSampler(dataset)
            shuffle = False

            if self.verbose:
                ngpu = dist.get_world_size()
                print(f"Device: CUDA ({ngpu} GPU{'' if ngpu == 1 else 's'})")
        else:
            if self.verbose:
                print(f"Device: {self.device}")

        # Create data loader
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
        )

        # Call on_train_begin
        for callback in callbacks:
            callback.on_train_begin()

        # Train
        operator.train()
        for epoch in range(epochs):
            loss_train = 0

            start = time()
            for x, u, y, v in data_loader:
                x, u = x.to(self.device), u.to(self.device)
                y, v = y.to(self.device), v.to(self.device)

                def closure(x=x, u=u, y=y, v=v):
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(operator, x, u, y, v)
                    loss.backward(retain_graph=True)
                    return loss

                self.optimizer.step(closure)

                # Compute mean loss
                loss_train += self.loss_fn(operator, x, u, y, v).detach().item()

            end = time()
            seconds_per_epoch = end - start
            loss_train /= len(data_loader)

            # Compute validation loss
            loss_val = None
            if val_dataset is not None:
                loss_val = dataset_loss(
                    val_dataset, operator, self.loss_fn, self.device
                )

            # Callbacks
            logs = Logs(
                epoch=epoch + 1,
                loss_train=loss_train,
                loss_val=loss_val,
                seconds_per_epoch=seconds_per_epoch,
            )

            for callback in callbacks:
                callback(logs)

            # Stopping criterion
            if criterion is not None:
                if criterion(logs):
                    break

        # Call on_train_end
        for callback in callbacks:
            callback.on_train_end()

        # Move operator back to CPU
        self.operator.to("cpu")
