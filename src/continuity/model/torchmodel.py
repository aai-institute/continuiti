import torch
from time import time
from continuity.model import device


class TorchModel(torch.nn.Module):
    """Torch model."""

    def compile(self, optimizer, criterion):
        """Compile model."""
        self.optimizer = optimizer
        self.criterion = criterion

        # Move to device
        self.to(device)

        # Print number of model parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {num_params}")

    def fit(self, dataset, epochs, writer=None):
        """Fit model to data set."""
        for epoch in range(epochs + 1):
            mean_loss = 0

            start = time()
            for i in range(len(dataset)):
                u, v, x = dataset[i]

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
