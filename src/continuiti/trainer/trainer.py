"""
`continuiti.trainer.trainer`
"""

import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, List, Union
from continuiti.data import OperatorDataset
from continuiti.operators import Operator
from continuiti.operators.losses import Loss, MSELoss
from continuiti.trainer.device import get_device
from .callbacks import Callback, PrintTrainingLoss
from .scheduler import LinearLRScheduler
from .criterion import Criterion, TrainingLossCriterion, TestLossCriterion
from .logs import Logs


class Trainer:
    """Trainer implements a default training loop for operator learning.

    Example:
        ```python
        from continuiti.trainer import Trainer
        from continuiti.operators.losses import MSELoss
        ...
        optimizer = torch.optim.Adam(operator.parameters(), lr=1e-3)
        loss_fn = MSELoss()
        trainer = Trainer(operator, optimizer, loss_fn, device="cuda:0")
        trainer.fit(dataset, tol=1e-3, epochs=1000)
        ```

    Args:
        operator: Operator to be trained.
        optimizer: Torch-like optimizer. Default is Adam with learning rate `lr`.
        lr: Learning rate. Ignored if optimizer is not None. Default is 1e-3.
        loss_fn: Loss function taking (op, x, u, y, v). Default is MSELoss.
        device: Device to train on. Default is CPU.
        verbose: Print model parameters and use PrintTrainingLoss callback by default. Default is True.
    """

    device = get_device()

    def __init__(
        self,
        operator: Operator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
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
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)

        # Verbosity
        if self.device.index is not None:
            if verbose is False:
                self.verbose = False
            else:
                self.verbose = self.device.index == 0
        else:
            self.verbose = verbose or True

    def fit(
        self,
        dataset: OperatorDataset,
        tol: float = 1e-5,
        epochs: int = 1000,
        callbacks: Optional[List[Callback]] = None,
        criterion: Optional[Criterion] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        test_dataset: Optional[OperatorDataset] = None,
        lr_scheduler: Union[bool, Callback] = True,
    ):
        """Fit operator to data set.

        Args:
            dataset: Data set.
            tol: Tolerance for stopping criterion. Ignored if criterion is not None.
            epochs: Maximum number of epochs.
            callbacks: List of additional callbacks.
            criterion: Stopping criterion. Defaults to TrainingLossCriteria(tol).
            batch_size: Batch size.
            shuffle: Shuffle data set.
            test_dataset: Test data set.
            lr_scheduler: Learning rate scheduler. If True, `LinearLRScheduler` is used.
        """
        # Callbacks
        callbacks = callbacks or []

        if self.verbose:
            steps = math.ceil(len(dataset) / batch_size)
            callbacks.append(PrintTrainingLoss(epochs, steps))

        if lr_scheduler is not False:
            if lr_scheduler is True:
                lr_scheduler = LinearLRScheduler(self.optimizer, epochs)
            callbacks.append(lr_scheduler)

        # Default criterion
        if criterion is None:
            if test_dataset is None:
                criterion = TrainingLossCriterion(tol)
            else:
                criterion = TestLossCriterion(tol)

        # Print number of model parameters
        if self.verbose:
            if hasattr(self.operator, "num_params"):
                num_params = self.operator.num_params()
            else:
                num_params = sum(p.numel() for p in self.operator.parameters())
            print(f"Parameters: {num_params}", end="  ")

        # Move operator to device
        operator = self.operator.to(self.device)

        # Use DistributedDataParallel if available
        is_distributed = dist.is_available() and dist.is_initialized()
        sampler, test_sampler = None, None
        if is_distributed:
            torch.cuda.set_device(self.device)

            operator = DDP(
                operator,
                device_ids=[self.device],
            )

            sampler = DistributedSampler(dataset, shuffle=shuffle)
            if test_dataset is not None:
                test_sampler = DistributedSampler(test_dataset, shuffle=shuffle)
            shuffle = False

            assert (
                batch_size % dist.get_world_size() == 0
            ), "Batch size must be divisible by world size"
            batch_size = batch_size // dist.get_world_size()  # Per-GPU batch size
            num_workers = dist.get_world_size()

            if self.verbose:
                ngpu = dist.get_world_size()
                print(f"Device: CUDA ({ngpu} GPU{'' if ngpu == 1 else 's'})")
        else:
            num_workers = 0
            if self.verbose:
                print(f"Device: {self.device}")

        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
        )
        if test_dataset is not None:
            test_data_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=test_sampler,
                num_workers=num_workers,
            )

        # Call on_train_begin
        for callback in callbacks:
            callback.on_train_begin()

        # Train
        loss_train, loss_test, epoch = None, None, 0
        for epoch in range(epochs):
            loss_train = 0

            if is_distributed:
                sampler.set_epoch(epoch)

            # Callbacks
            logs = Logs(
                epoch=epoch + 1,
                step=0,
                loss_train=loss_train,
                loss_test=loss_test,
            )

            operator.train()
            for xuyv in data_loader:
                xuyv = [t.to(self.device) for t in xuyv]

                def closure(xuyv=xuyv):
                    self.optimizer.zero_grad()
                    loss = self.loss_fn(operator, *xuyv)
                    loss.backward(retain_graph=True)
                    return loss

                loss = self.optimizer.step(closure)

                # Compute mean loss
                loss_train += loss.detach().item()

                # Callbacks
                logs.step += 1
                logs.loss_train = loss_train / logs.step

                for callback in callbacks:
                    callback.step(logs)

            # Compute test loss
            if test_dataset is not None:
                operator.eval()
                loss_test = 0
                for xuyv in test_data_loader:
                    xuyv = [t.to(self.device) for t in xuyv]
                    loss = self.loss_fn(operator, *xuyv)
                    if is_distributed:
                        dist.all_reduce(loss)
                        loss /= dist.get_world_size()
                    loss_test += loss.detach().item()
                loss_test /= len(test_data_loader)

            logs.loss_test = loss_test

            # Callbacks
            for callback in callbacks:
                callback(logs)

            # Stopping criterion
            if criterion is not None:
                if criterion(logs):
                    if self.verbose:
                        print("- stopping criterion met")
                    break

        # Call on_train_end
        for callback in callbacks:
            callback.on_train_end()

        # Move operator back to CPU
        self.operator.to("cpu")

        return logs
