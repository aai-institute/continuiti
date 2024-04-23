import torch.nn as nn
import torch
import pytest
from continuiti.networks import FullyConnected


@pytest.fixture(scope="session")
def trivial_fully_connected():
    return FullyConnected(input_size=3, output_size=5, width=7)


@pytest.fixture(scope="session")
def random_vector():
    return torch.rand(
        3,
    )


class TestFullyConnected:
    def test_can_initialize(self, trivial_fully_connected):
        assert isinstance(trivial_fully_connected, FullyConnected)

    def test_can_forward(self, trivial_fully_connected, random_vector):
        trivial_fully_connected(random_vector)
        assert True

    def test_shape_correct(self, trivial_fully_connected, random_vector):
        out = trivial_fully_connected(random_vector)
        assert out.shape == torch.Size([5])

    def test_can_backward(self, trivial_fully_connected, random_vector):
        out = trivial_fully_connected(random_vector)
        loss = nn.L1Loss()(
            out,
            torch.rand(
                5,
            ),
        )
        loss.backward()
        assert True

    def test_can_overfit(self, trivial_fully_connected, random_vector):
        out_vec = torch.rand(
            5,
        )
        criterion = nn.L1Loss()
        optim = torch.optim.Adam(trivial_fully_connected.parameters())

        loss = torch.inf
        for _ in range(1000):
            optim.zero_grad()
            out = trivial_fully_connected(random_vector)
            loss = criterion(out, out_vec)
            loss.backward()
            if loss.item() <= 1e-3:
                break
            optim.step()

        assert loss.item() <= 1e-3
