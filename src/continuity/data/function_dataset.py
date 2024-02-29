import torch
from typing import Optional
from continuity.discrete.sampler import Sampler
from .function import SampledFunctionSet, FunctionSet
from .dataset import OperatorDataset


class FunctionOperatorDataset(OperatorDataset):
    """A dataset class for generating samples from function sets.

    This class extends `OperatorDataset` to specifically handle scenarios where both inputs and outputs are functions.
    It utilizes samplers to generate discrete representations of function spaces (input and solution function sets) and
    physical spaces.


    Args:
        input_function_set: A function set representing the input.
        x_sampler: A sampler for generating discrete representations of the input function in space.
        n_sensors: The number of sensors to sample in the input physical space.
        solution_function_set: A function set representing the solution set of the underlying operator.
        y_sampler: A sampler for generating discrete space representations of the solution space.
        n_evaluations: The number of evaluation points to sample in the solution physical space.
        p_sampler: A sampler for sampling parameters to instantiate functions from the function sets.
        n_observations: The number of observations to generate.
        x_transform, u_transform, y_transform, v_transform: Optional transformation functions applied to the
            sampled physical spaces and function evaluations, respectively.
    """

    def __init__(
        self,
        input_function_set: FunctionSet,
        x_sampler: Sampler,
        n_sensors: int,
        solution_function_set: FunctionSet,
        y_sampler: Sampler,
        n_evaluations: int,
        p_sampler: Sampler,
        n_observations: int,
        x_transform: Optional,
        u_transform: Optional,
        y_transform: Optional,
        v_transform: Optional,
    ):
        # sample the parameter space
        p_samples = p_sampler(n_observations)

        # sample function spaces
        sampled_input_space = SampledFunctionSet(
            function_set=input_function_set, samples=p_samples
        )
        sampled_solution_space = SampledFunctionSet(
            function_set=solution_function_set, samples=p_samples
        )

        # sample physical spaces
        x = torch.stack([x_sampler(n_sensors) for _ in range(n_observations)])
        y = torch.stack([y_sampler(n_evaluations) for _ in range(n_observations)])

        # create ground truth data
        u = sampled_input_space(x)
        v = sampled_solution_space(y)

        super().__init__(
            x=x,
            u=u,
            y=y,
            v=v,
            x_transform=x_transform,
            u_transform=u_transform,
            y_transform=y_transform,
            v_transform=v_transform,
        )
