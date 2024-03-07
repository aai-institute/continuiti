"""
`continuity.data.function_dataset.function_dataset`

Function data set implementations.
"""

import torch
from typing import Optional, Tuple
from continuity.discrete.sampler import Sampler
from continuity.data import OperatorDataset
from continuity.data.function_dataset.function_set import FunctionSet


class FunctionOperatorDataset(OperatorDataset):
    """A dataset class for generating samples from function sets.

    This class extends `OperatorDataset` to specifically handle scenarios where both inputs and outputs are known
    functions. It utilizes samplers to generate discrete representations of function spaces (input and solution function
    sets) and physical spaces.

    Args:
        input_function_set: A function set representing the input space.
        x_sampler: A sampler for generating discrete representations of the domain.
        n_sensors: The number of sensors to sample in the input physical space.
        solution_function_set: A function set representing the solution set of the underlying operator.
        y_sampler: A sampler for generating discrete representations of the codomain.
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
        x_transform: Optional = None,
        u_transform: Optional = None,
        y_transform: Optional = None,
        v_transform: Optional = None,
    ):
        self.input_function_set = input_function_set
        self.x_sampler = x_sampler
        self.solution_function_set = solution_function_set
        self.y_sampler = y_sampler
        self.p_sampler = p_sampler

        x, u, y, v = self.generate_observations(
            n_sensors=n_sensors,
            n_evaluations=n_evaluations,
            n_observations=n_observations,
        )

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

    def generate_observations(
        self,
        n_sensors: int,
        n_evaluations: int,
        n_observations: int,
        x_sampler: Sampler = None,
        y_sampler: Sampler = None,
        p_sampler: Sampler = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates new observations of the domain, input function, codomain, and solution function.

        Args:
            n_sensors: The number of sensors to sample in the input physical space.
            n_evaluations: The number of evaluation points to sample in the solution physical space.
            n_observations: The number of observations to generate.
            x_sampler: A sampler for generating discrete representations of the input function in space.
            y_sampler: A sampler for generating discrete representations of the solution function in space.
            p_sampler: A sampler for sampling parameters to instantiate functions from the function sets.

        Returns:
            Tuple containing the space samples $x$ of the input space, the evaluated input function $u$, the space
                samples $y$ of the solution space, and the evaluated solution function $v$.
        """
        if x_sampler is None:
            x_sampler = self.x_sampler
        if y_sampler is None:
            y_sampler = self.y_sampler
        if p_sampler is None:
            p_sampler = self.p_sampler

        # sample the parameter space
        p_samples = p_sampler(n_observations)

        # sample physical spaces
        x = torch.stack([x_sampler(n_sensors) for _ in range(n_observations)])
        y = torch.stack([y_sampler(n_evaluations) for _ in range(n_observations)])

        # create ground truth data
        u = self.input_function_set(parameters=p_samples, locations=x)
        v = self.solution_function_set(parameters=p_samples, locations=y)

        return x, u, y, v
