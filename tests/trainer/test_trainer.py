import torch
import pytest
from torch.utils.data import DataLoader
from continuity.operators import DeepONet
from continuity.data.sine import Sine
from continuity.trainer import Trainer
from continuity.trainer import Trainer
import torch.distributed as dist

torch.manual_seed(0)


def train(rank: int = "cpu", verbose: bool = True):
    dataset = Sine(num_sensors=32, size=16)
    data_loader = DataLoader(dataset)
    operator = DeepONet(dataset.shapes)

    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-2)
    trainer = Trainer(operator, optimizer, device=rank, verbose=verbose)
    trainer.fit(data_loader, epochs=2)

    # Make sure we can use operator output on cpu again
    x, u, y, v = next(iter(data_loader))
    v_pred = operator(x, u, y)
    mse = ((v_pred - v.to("cpu")) ** 2).mean()
    if verbose:
        print(f"mse = {mse.item():.3g}")


@pytest.mark.slow
def test_trainer():
    train()


# Use ./run_parallel.sh to run test with CUDA
def train_parallel():
    if torch.cuda.device_count() == 0:
        print("Skipping CUDA tests because no GPU is available.")
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()

    if rank == 0:
        print(f" == Number of GPUs: {dist.get_world_size()}")

    verbose = rank == 0
    train(rank, verbose=verbose)

    dist.destroy_process_group()


if __name__ == "__main__":
    train_parallel()
