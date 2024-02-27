import pytest
from continuity.operators import DeepONet
from continuity.data.sine import Sine
from continuity.trainer import Trainer


def train():
    dataset = Sine(num_sensors=32, size=256)
    operator = DeepONet(dataset.shapes)

    trainer = Trainer(operator)
    trainer.fit(dataset, epochs=100)

    # Make sure we can use operator output on cpu again
    x, u, y, v = dataset[0]
    v_pred = operator(x.unsqueeze(0), u.unsqueeze(0), y.unsqueeze(0)).squeeze(0)
    assert ((v_pred - v.to("cpu")) ** 2).mean() < 0.1


@pytest.mark.slow
def test_trainer():
    train()


# Use ./run_parallel.sh to run test with CUDA
if __name__ == "__main__":
    train()
