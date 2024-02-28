import torch

from continuity.discrete import (
    Sampler,
    BoxSampler,
    UniformBoxSampler,
    RegularGridSampler,
)
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


class FunctionOperatorDatasetFactory:
    def __init__(self, p_sampler: type(BoxSampler), x_sampler: type(BoxSampler)):
        self.p_sampler = p_sampler
        self.x_sampler = x_sampler

    def __call__(
        self,
        function_space: FunctionSpace,
        p_min: torch.Tensor,
        p_max: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        n_observations: int,
        n_sensors: int,
    ) -> FunctionOperatorDataset:
        """

        Args:
            function_space:
            p_min:
            p_max:
            x_min:
            x_max:

        Returns:

        """
        p_sampler = self.p_sampler(x_min=p_min, x_max=p_max)
        x_sampler = self.x_sampler(x_min=x_min, x_max=x_max)

        return FunctionOperatorDataset(
            function_space=function_space,
            p_sampler=p_sampler,
            x_sampler=x_sampler,
            n_observations=n_observations,
            n_sensors=n_sensors,
        )


class DisInvSamInvFunctionOperatorDatasetFactory(FunctionOperatorDatasetFactory):
    """
    Discretization Invariant and Sample Invariant
    """

    def __init__(self, p_sampler: type(BoxSampler), x_sampler: type(BoxSampler)):
        super().__init__(p_sampler, x_sampler)
        raise NotImplementedError("Necessary sampler has not been implemented yet!")


class DisInvSamUniFunctionOperatorDatasetFactory(FunctionOperatorDatasetFactory):
    """
    Discretization Invariant and Sample Uniform.
    """

    def __init__(self, p_sampler: type(BoxSampler)):
        x_sampler = UniformBoxSampler
        super().__init__(p_sampler, x_sampler)


class DisUniSamUniFunctionOperatorDatasetFactory(FunctionOperatorDatasetFactory):
    """
    Discretization Uniform and Sample Uniform.
    """

    def __init__(self, p_sampler: type(BoxSampler)):
        x_sampler = RegularGridSampler
        super().__init__(p_sampler, x_sampler)
