import torch.nn as nn
import torch
import pytest
from continuity.networks import DeepResidualNetwork


@pytest.fixture(scope="session")
def trivial_res_net():
    return DeepResidualNetwork(3, 5, 7, 2, act=nn.GELU())


@pytest.fixture(scope="session")
def random_vector():
    return torch.rand(
        3,
    )


class TestResNet:
    def test_can_initialize(self, trivial_res_net):
        assert isinstance(trivial_res_net, DeepResidualNetwork)

    def test_can_forward(self, trivial_res_net, random_vector):
        trivial_res_net(random_vector)
        assert True

    def test_shape_correct(self, trivial_res_net, random_vector):
        out = trivial_res_net(random_vector)
        assert out.shape == torch.Size([5])

    def test_can_backward(self, trivial_res_net, random_vector):
        out = trivial_res_net(random_vector)
        loss = nn.L1Loss()(
            out,
            torch.rand(
                5,
            ),
        )
        loss.backward()
        assert True

    def test_can_overfit(self, trivial_res_net, random_vector):
        out_vec = torch.rand(
            5,
        )
        criterion = nn.L1Loss()
        optim = torch.optim.Adam(trivial_res_net.parameters())

        loss = torch.inf
        for _ in range(1000):
            optim.zero_grad()
            out = trivial_res_net(random_vector)
            loss = criterion(out, out_vec)
            loss.backward()
            optim.step()

        assert loss.item() <= 1e-3
