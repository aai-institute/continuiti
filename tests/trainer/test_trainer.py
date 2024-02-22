import torch
from torch.utils.data import DataLoader
from continuity.operators import DeepONet
from continuity.data import device
from continuity.data.sine import Sine
from continuity.trainer import Trainer

torch.manual_seed(0)


def test_trainer():
    dataset = Sine(num_sensors=32, size=16)
    data_loader = DataLoader(dataset)
    operator = DeepONet(dataset.shapes)

    print(f"Using device: {device}")
    trainer = Trainer(operator, device=device)
    trainer.fit(data_loader, epochs=2)

    # Make sure we can use operator output on cpu again
    x, u, y, v = next(iter(data_loader))
    v_pred = operator(x, u, y)
    mse = ((v_pred - v.to("cpu")) ** 2).mean()
    print(f"mse = {mse.item():.3g}")


if __name__ == "__main__":
    test_trainer()
