import torch

from continuity.discrete import Sampler
from .function import SampledFunctionSpace, FunctionSpace
from .dataset import OperatorDataset


class FunctionOperatorDataset(OperatorDataset):
    def __init__(
        self,
        function_space: FunctionSpace,
        p_sampler: Sampler,
        x_sampler: Sampler,
        n_observations: int,
        n_sensors: int,
    ):
        # sample function space
        sampled_space = SampledFunctionSpace(
            function_space=function_space,
            sampler=p_sampler,
            n_observations=n_observations,
        )

        # sample physical space
        x = torch.stack([x_sampler(n_sensors) for _ in range(n_observations)])

        # create ground truth data
        u = sampled_space(x)

        super().__init__(x, u, x, u)
