import torch
from typing import List, Self


class Function:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapping(x)

    def __add__(self, other: Self) -> Self:
        return Function(mapping=lambda x: self.mapping(x) + other.mapping(x))

    def __mul__(self, other: Self) -> Self:
        return Function(mapping=lambda x: self.mapping(x) * other.mapping(x))


class ParameterizedFunction:
    def __init__(self, mapping, n_parameters: int, parameters_dtype: List):
        self.mapping = mapping
        self.n_parameters = n_parameters
        self.parameters_dtype = parameters_dtype

    def __call__(self, parameters: torch.Tensor) -> List[Function]:
        return [Function(lambda x, a=param: self.mapping(a, x)) for param in parameters]

    def __add__(self, other: Self) -> Self:
        n_parameters = self.n_parameters + other.n_parameters
        parameters_dtype = self.parameters_dtype + other.parameters_dtype
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            + other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
            parameters_dtype=parameters_dtype,
        )

    def __mul__(self, other: Self) -> Self:
        n_parameters = self.n_parameters + other.n_parameters
        parameters_dtype = self.parameters_dtype + other.parameters_dtype
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            * other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
            parameters_dtype=parameters_dtype,
        )
