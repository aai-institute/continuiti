import torch
import pytest
from torch.utils.data import DataLoader
from continuity.operators import DeepONet
from continuity.data import Sine
from continuity.trainer import Trainer, Timer
import torch.distributed as dist

torch.manual_seed(0)


def train(rank: int = 0):
    dataset = Sine(num_sensors=32, size=128)
    data_loader = DataLoader(dataset, batch_size=128)

    operator = DeepONet(dataset.shapes)

    optimizer = torch.optim.Adam(operator.parameters(), lr=1e-2)
    trainer = Trainer(operator, optimizer, device=rank)

    with Timer(rank):
        trainer.fit(data_loader, epochs=10)


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

    train(rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    train_parallel()
