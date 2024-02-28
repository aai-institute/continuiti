from typing import List, Self

import torch

from continuity.discrete import Sampler
from .function import ParameterizedFunction, Function


class FunctionSpace:
    def __init__(self, function: ParameterizedFunction):
        self.function = function

    def __call__(self, parameters) -> List[Function]:
        return self.function(parameters)

    def __add__(self, other: Self) -> Self:
        function = self.function + other.function
        return FunctionSpace(function=function)

    def __mul__(self, other: Self) -> Self:
        function = self.function * other.function
        return FunctionSpace(function=function)


class SampledFunctionSpace:
    def __init__(
        self, function_space: FunctionSpace, sampler: Sampler, n_observations: int
    ):
        samples = sampler(n_observations)
        self.functions = function_space(samples)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        observations = [fun(xi) for fun, xi in zip(self.functions, x)]
        return torch.stack(observations)
