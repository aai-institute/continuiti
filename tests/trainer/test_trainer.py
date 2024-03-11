import pytest
from continuity.operators import DeepONet
from continuity.data.sine import Sine
from continuity.trainer import Trainer


def train():
    dataset = Sine(num_sensors=32, size=256)
    operator = DeepONet(dataset.shapes, trunk_depth=16)

    Trainer(operator).fit(dataset, tol=1e-3)

    # Make sure we can use operator output on cpu again
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    v_pred = operator(x, u, y)
    assert ((v_pred - v.to("cpu")) ** 2).mean() < 1e-3


@pytest.mark.slow
def test_trainer():
    train()


# Use ./run_parallel.sh to run test with CUDA
if __name__ == "__main__":
    train()
