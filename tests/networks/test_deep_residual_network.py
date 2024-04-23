import torch.nn as nn
import torch
import pytest
from continuiti.networks import DeepResidualNetwork


@pytest.fixture(scope="session")
def trivial_deep_residual_network():
    return DeepResidualNetwork(input_size=3, output_size=5, width=15, depth=3)


@pytest.fixture(scope="session")
def random_vector():
    return torch.rand(
        3,
    )


class TestDeepResidualNetwork:
    def test_can_initialize(self, trivial_deep_residual_network):
        assert isinstance(trivial_deep_residual_network, DeepResidualNetwork)

    def test_can_forward(self, trivial_deep_residual_network, random_vector):
        trivial_deep_residual_network(random_vector)
        assert True

    def test_shape_correct(self, trivial_deep_residual_network, random_vector):
        out = trivial_deep_residual_network(random_vector)
        assert out.shape == torch.Size([5])

    def test_can_backward(self, trivial_deep_residual_network, random_vector):
        out = trivial_deep_residual_network(random_vector)
        loss = nn.L1Loss()(
            out,
            torch.rand(
                5,
            ),
        )
        loss.backward()
        assert True

    def test_can_overfit(self, trivial_deep_residual_network, random_vector):
        out_vec = torch.rand(
            5,
        )
        criterion = nn.L1Loss()
        optim = torch.optim.Adam(trivial_deep_residual_network.parameters(), lr=1e-4)

        loss = torch.inf
        for _ in range(1000):
            optim.zero_grad()
            out = trivial_deep_residual_network(random_vector)
            loss = criterion(out, out_vec)
            loss.backward()
            if loss.item() <= 1e-3:
                break
            optim.step()

        assert loss.item() <= 1e-3
