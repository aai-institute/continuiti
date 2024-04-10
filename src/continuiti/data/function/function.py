"""
`continuiti.data.function.function`

Function.
"""
from __future__ import annotations
import torch
from typing import Callable


class Function:
    r"""A class for creating and manipulating functions.

    A function is a mapping between a domain X (supported on a field K) and a codomain Y (supported on a field F)
    denoted by
    $$
    f: X \rightarrow Y, x \mapsto f(x).
    $$
    For this class, scalar multiplication and addition with other function instances are implemented. This is done to
    ensure that function instances are able to fulfill properties needed to create a function space (vector space
    over a function set).

    Args:
        mapping: A callable that accepts exactly one argument, a torch.Tensor, and returns a torch.Tensor. This
        callable represents the mathematical function to be applied on a PyTorch tensor.

    Example:
        ```python
        f = Function(lambda x: x**2)
        g = Function(lambda x: x + 1)
        h = f + 2 * g
        x = torch.tensor(2.0)
        h(x)
        ```
        Output:
        ```shell
        tensor(10.)
        ```
    """

    def __init__(self, mapping: Callable):
        self.mapping = mapping

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Evaluates the encapsulated function.

        Args:
            *args: list of arguments passed to the mapping class attribute.
            **kwargs: dict of arguments passed to the mapping class attribute.

        Returns:
            The result of applying the function to all arguments.
        """
        return self.mapping(*args, **kwargs)

    def __add__(self, other: Function) -> Function:
        """Creates a new Function representing the addition of this function with another.

        Args:
            other: Another Function instance to add to this one. The other function needs to have the same arguments as
                this one.

        Returns:
            A new Function instance representing the addition of the two functions.
        """
        return Function(mapping=lambda args: self.mapping(args) + other.mapping(args))

    def __mul__(self, scalar: float) -> Function:
        """Creates a new Function representing the multiplication of this function with a scalar.

        Args:
            scalar: Scalar to multiply this function with.

        Returns:
            A new Function instance representing the multiplication of this function with the scalar.
        """
        return Function(mapping=lambda args: scalar * self.mapping(args))

    def __rmul__(self, scalar: float) -> Function:
        return self.__mul__(scalar)
