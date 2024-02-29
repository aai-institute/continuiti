"""
`continuity.data.function.parameterized_function`

Parameterized function implementation.
"""
from __future__ import annotations
import torch
from typing import List, Callable
from .function import Function


class ParameterizedFunction:
    """
    A class for creating and manipulating parameterized functions using PyTorch tensors.

    This class allows for the encapsulation of mathematical functions that not only operate on
    PyTorch tensors but are also parameterized by a set of parameters. It supports basic arithmetic
    operations (addition, subtraction, multiplication, and division) between instances, properly
    handling parameter updates.

    Args:
        mapping: A callable that accepts exactly two arguments, two torch.Tensors, and returns a torch.Tensor. This
            callable represents the mathematical parameterized function to be applied on a PyTorch tensor `x` with
            variable parameters `param`.
        n_parameters: The number of parameters that the `mapping` function expects.
        parameters_dtype: A list specifying the data types of the parameters, which aids in
            the construction and manipulation of parameters.
    """

    def __init__(self, mapping: Callable, n_parameters: int):
        self.mapping = mapping
        self.n_parameters = n_parameters

    def __call__(self, parameters: torch.Tensor) -> List[Function]:
        """Applies the parameterized function to the given parameters, returning a list of Function instances.

        Args:
            parameters: A tensor containing the parameters to be used with the mapping function.

        Returns:
            A list of Function instances, each parameterized by a single element from the
            `parameters` tensor.
        """
        return [Function(lambda x, a=param: self.mapping(a, x)) for param in parameters]

    def __get_operator_parameter_update(self, other: ParameterizedFunction) -> int:
        """
        Private method to calculate the updated number of parameters and their data types after an arithmetic operation.

        Args:
            other: Another ParameterizedFunction instance to perform the operation with.

        Returns:
            Integer representing the new number of parameters. The order in which the parameters are updated follows
                self and then the others' parameters.
        """
        n_parameters = self.n_parameters + other.n_parameters
        return n_parameters

    def __add__(self, other: ParameterizedFunction) -> ParameterizedFunction:
        """Creates a new ParameterisedFunction representing the addition of this parameterized function
        with another parameterized function.

        Args:
            other: Another ParameterizedFunction instance to add to this one.

        Returns:
            A new Function instance representing the addition of the two functions.
        """
        n_parameters = self.__get_operator_parameter_update(other)
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            + other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
        )

    def __sub__(self, other: ParameterizedFunction) -> ParameterizedFunction:
        """Creates a new ParameterisedFunction representing the subtraction of another parameterized function from this
        parameterized function.

        Args:
            other: Another ParameterizedFunction instance to subtract from this one.

        Returns:
            A new Function instance representing the subtraction of the two functions.
        """

        n_parameters = self.__get_operator_parameter_update(other)
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            - other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
        )

    def __mul__(self, other: ParameterizedFunction) -> ParameterizedFunction:
        """Creates a new Function representing the multiplication of this parameterized function with another.

        Args:
            other: Another ParameterizedFunction instance to multiply with this one.

        Returns:
            A new ParameterizedFunction instance representing the multiplication of the two functions.
        """
        n_parameters = self.__get_operator_parameter_update(other)
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            * other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
        )

    def __truediv__(self, other: ParameterizedFunction) -> ParameterizedFunction:
        """Creates a new ParameterizedFunction representing the division of this parameterized function by another.

        Args:
            other: Another ParameterizedFunction instance to divide this one by.

        Returns:
            A new ParameterizedFunction instance representing the division of the two functions.
        """
        n_parameters = self.__get_operator_parameter_update(other)
        return ParameterizedFunction(
            mapping=lambda a, x: self.mapping(a[: self.n_parameters], x)
            / other.mapping(a[self.n_parameters :], x),
            n_parameters=n_parameters,
        )
