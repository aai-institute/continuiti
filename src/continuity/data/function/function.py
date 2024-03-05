"""
`continuity.data.function.function`

Function.
"""
from __future__ import annotations
import torch
from typing import Callable


class Function:
    """A class for creating and manipulating functions.

    This class allows for the encapsulation of arbitrary mathematical functions that operate on
    PyTorch tensors, providing an interface to perform basic arithmetic operations between functions
    such as addition, subtraction, multiplication, and division.

    Args:
        mapping: A callable that accepts exactly one argument, a torch.Tensor, and returns a torch.Tensor. This
        callable represents the mathematical function to be applied on a PyTorch tensor.

    Example:
        ```python
        f = Function(lambda x: x**2)
        g = Function(lambda x: x + 1)
        h = f + g
        x = torch.tensor(2.0)
        h(x)
        ```
        Output:
        ```shell
        tensor(7.)
        ```
    """

    def __init__(self, mapping: Callable):
        self.mapping = mapping

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the encapsulated mathematical function on a given tensor.

        Args:
            x: The input tensor to apply the function to.

        Returns:
            The result of applying the function to `x`.
        """
        return self.mapping(x)

    def __add__(self, other: Function) -> Function:
        """Creates a new Function representing the addition of this function with another.

        Args:
            other: Another Function instance to add to this one.

        Returns:
            A new Function instance representing the addition of the two functions.
        """
        return Function(mapping=lambda x: self.mapping(x) + other.mapping(x))

    def __sub__(self, other: Function) -> Function:
        """
        Creates a new Function representing the subtraction of another function from this one.

        Args:
            other: Another Function instance to subtract from this one.

        Returns:
            A new Function instance representing the subtraction of the two functions.
        """
        return Function(mapping=lambda x: self.mapping(x) - other.mapping(x))

    def __mul__(self, other: Function) -> Function:
        """Creates a new Function representing the multiplication of this function with another.

        Args:
            other: Another Function instance to multiply with this one.

        Returns:
            A new Function instance representing the multiplication of the two functions.
        """
        return Function(mapping=lambda x: self.mapping(x) * other.mapping(x))

    def __truediv__(self, other: Function) -> Function:
        """Creates a new Function representing the division of this function by another.

        Args:
            other: Another Function instance to divide this one by.

        Returns:
            A new Function instance representing the division of the two functions.
        """
        return Function(mapping=lambda x: self.mapping(x) / other.mapping(x))
