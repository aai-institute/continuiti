from typing import List, Self
from .function import ParameterizedFunction, Function


class FunctionSet:
    """A class to represent a set of functions encapsulated within a ParameterizedFunction instance.

    This class provides a way to apply a parameterized function across a set of parameters, effectively
    managing a collection of functions that can be evaluated, manipulated, or combined. It supports
    basic arithmetic operations add and sub with other FunctionSet instances by performing the operation on their
    underlying ParameterizedFunction objects.

    Args:
        function: The ParameterizedFunction instance that defines the set of functions.

    Example:
        ```python
        # Assuming `pf` is a previously defined ParameterizedFunction instance
        fs1 = FunctionSet(function=pf)
        fs2 = FunctionSet(function=pf)
        fs3 = fs1 + fs2
        ```
        This example demonstrates creating two FunctionSet instances from a ParameterizedFunction `pf`
        and then combining them using the addition operation, resulting in a new FunctionSet `fs3`.
    """

    def __init__(self, function: ParameterizedFunction):
        self.function = function

    def __call__(self, parameters) -> List[Function]:
        """
        Applies the encapsulated ParameterizedFunction to the given parameters, returning a list of Function instances.

        Args:
            parameters: The parameters to be used with the ParameterizedFunction. The specific type and
            structure of `parameters` depend on the requirements of the encapsulated ParameterizedFunction.

        Returns:
            List[Function]: A list of Function instances resulting from applying the ParameterizedFunction
            to the `parameters`.
        """
        return self.function(parameters)

    def __add__(self, other: Self) -> Self:
        """Combines this FunctionSet with another by adding their underlying ParameterizedFunction objects,
        returning a new FunctionSet instance.

        Args:
            other: Another FunctionSet instance to add to this one.

        Returns:
            FunctionSet: A new FunctionSet instance representing the addition of this and the `other` FunctionSet.
        """
        function = self.function + other.function
        return FunctionSet(function=function)

    def __sub__(self, other: Self) -> Self:
        """Combines this FunctionSet with another by subtracting their underlying ParameterizedFunction objects,
        returning a new FunctionSet instance.

        Args:
            other: Another FunctionSet instance to subtract from this one.

        Returns:
            FunctionSet: A new FunctionSet instance representing the subtraction of this and the `other` FunctionSet.
        """
        function = self.function - other.function
        return FunctionSet(function=function)
