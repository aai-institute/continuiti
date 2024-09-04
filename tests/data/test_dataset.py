import pytest
from typing import Tuple, List
import torch
from random import randint
from torch.utils.data import DataLoader
from continuiti.data.selfsupervised import SelfSupervisedOperatorDataset
from continuiti.data import MaskedOperatorDataset
from continuiti.transforms import Normalize


def test_dataset():
    # Sensors
    num_sensors = 4
    f = lambda x: x**2
    num_channels = 1
    coordinate_dim = 1

    x = torch.Tensor(range(num_sensors)).reshape(-1, 1)
    u = f(x)

    # Dataset
    dataset = SelfSupervisedOperatorDataset(x.unsqueeze(0), u.unsqueeze(0))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    x_target, u_target = x, u

    # Test
    for x, u, y, v in dataloader:
        # Every x, u must equal observation
        assert x.shape[1] == num_sensors
        assert u.shape[1] == num_sensors
        assert x.shape[2] == coordinate_dim
        assert u.shape[2] == num_channels
        assert (x == x_target).all()
        assert (u == u_target).all()

        # Every y, v must equal f(y)
        assert y.shape[1] == coordinate_dim
        assert v.shape[1] == num_channels
        assert (v == f(y)).all()


class TestMaskedDataset:
    observations = 19
    batch_size = 2
    x_dim = 3
    u_dim = 5
    y_dim = 7
    v_dim = 11
    max_sensors = 13
    max_evaluations = 17

    @pytest.fixture
    def random_masked_tensors(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x = torch.rand(self.observations, self.x_dim, self.max_sensors)
        u = torch.rand(self.observations, self.u_dim, self.max_sensors)
        ipt_mask = torch.rand(self.observations, self.max_sensors) > 0.25

        y = torch.rand(self.observations, self.y_dim, self.max_evaluations)
        v = torch.rand(self.observations, self.v_dim, self.max_evaluations)
        opt_mask = torch.rand(self.observations, self.max_evaluations) > 0.25

        return x, u, ipt_mask, y, v, opt_mask

    @pytest.fixture
    def random_lists(
        self,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
    ]:
        n_sensors = [randint(1, self.max_sensors) for _ in range(self.observations)]
        x = [torch.rand(self.x_dim, n) for n in n_sensors]
        u = [torch.rand(self.u_dim, n) for n in n_sensors]

        n_observations = [
            randint(1, self.max_evaluations) for _ in range(self.observations)
        ]
        y = [torch.rand(self.y_dim, n) for n in n_observations]
        v = [torch.rand(self.v_dim, n) for n in n_observations]

        return x, u, y, v

    @pytest.fixture
    def random_masked_dataset(self, random_masked_tensors):
        x, u, ipt_mask, y, v, opt_mask = random_masked_tensors
        return MaskedOperatorDataset(
            x=x, u=u, y=y, v=v, ipt_mask=ipt_mask, opt_mask=opt_mask
        )

    def test_can_initialize_tensor(self, random_masked_tensors):
        x, u, ipt_mask, y, v, opt_mask = random_masked_tensors
        dataset = MaskedOperatorDataset(
            x=x, u=u, y=y, v=v, ipt_mask=ipt_mask, opt_mask=opt_mask
        )
        assert isinstance(dataset, MaskedOperatorDataset)

    def test_can_initalize_list(self, random_lists):
        x, u, y, v = random_lists
        dataset = MaskedOperatorDataset(x=x, u=u, y=y, v=v)
        assert isinstance(dataset, MaskedOperatorDataset)

    def test_dataset_length(self, random_masked_dataset):
        assert len(random_masked_dataset) == self.observations

    def test_masking(self, random_masked_dataset, random_masked_tensors):
        _, _, ipt_mask, _, _, opt_mask = random_masked_tensors

        for (
            (
                _,
                _,
                _,
                _,
                ipt_mask_i,
                opt_mask_i,
            ),
            ipt_mask_gt,
            opt_mask_gt,
        ) in zip(iter(random_masked_dataset), ipt_mask, opt_mask):
            assert torch.all(ipt_mask_i == ipt_mask_gt)
            assert torch.all(opt_mask_i == opt_mask_gt)

    def test_unmasked_samples(self, random_masked_dataset):
        for x, u, y, v, _, _ in random_masked_dataset:
            assert not torch.allclose(x, torch.zeros(x.shape))
            assert not torch.allclose(u, torch.zeros(u.shape))
            assert not torch.allclose(y, torch.zeros(y.shape))
            assert not torch.allclose(v, torch.zeros(v.shape))

    def test_batch_masking(self, random_masked_dataset):
        dataloader = DataLoader(
            random_masked_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True,
        )

        for x, u, y, v, ipt_mask, opt_mask in dataloader:
            assert (
                len(x)
                == len(u)
                == len(y)
                == len(v)
                == len(ipt_mask)
                == len(opt_mask)
                == self.batch_size
            )

    def test_masked_and_transform(self, random_masked_tensors):
        x, u, ipt_mask, y, v, opt_mask = random_masked_tensors
        dataset = MaskedOperatorDataset(
            x=x,
            u=u,
            y=y,
            v=v,
            ipt_mask=ipt_mask,
            opt_mask=opt_mask,
            x_transform=Normalize(
                torch.ones(self.x_dim, 1) / 2.0,
                torch.ones(self.x_dim, 1) * torch.sqrt(torch.tensor([1 / 12])),
            ),
            u_transform=Normalize(
                torch.ones(self.u_dim, 1) / 2.0,
                torch.ones(self.u_dim, 1) * torch.sqrt(torch.tensor([1 / 12])),
            ),
            y_transform=Normalize(
                torch.ones(self.y_dim, 1) / 2.0,
                torch.ones(self.y_dim, 1) * torch.sqrt(torch.tensor([1 / 12])),
            ),
            v_transform=Normalize(
                torch.ones(self.v_dim, 1) / 2.0,
                torch.ones(self.v_dim, 1) * torch.sqrt(torch.tensor([1 / 12])),
            ),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        for x, u, y, v, ipt_mask, opt_mask in dataloader:
            assert True
