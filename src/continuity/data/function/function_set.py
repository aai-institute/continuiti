from typing import List, Self

import torch

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


class SampledFunctionSet:
    """
    Represents a set of functions that have been sampled from a specified function space, allowing for
    the evaluation of these functions at different points.

    This class takes a `FunctionSet`, representing a space of parameterized functions, and a set of samples
    (parameters), creating a collection of sampled functions. These sampled functions can then be evaluated
    at given input points, producing a tensor of observations.

    Args:
        function_set: A function set from which the samples for this `SampledFunctionSet` will be drawn.
        samples: Parameter samples, where each row represents one set of parameters to evaluate the encapsulated
            parameterized function.

    Attributes:
        functions: A list of `Function` instances, each sampled from the `function_set` with the given `samples`.
    """

    def __init__(self, function_set: FunctionSet, samples: torch.Tensor):
        self.functions = function_set(samples)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates all sampled functions at the corresponding points in `x`.

        This method iterates over the sampled functions and the rows of input tensor `x`, evaluating each function
        at its corresponding point in `x`. The results are then stacked into a single tensor, representing
        the observations from evaluating the set of sampled functions.

        Args:
            x: A tensor containing points at which to evaluate the sampled functions.

        Returns:
            A tensor containing the observations from evaluating each sampled function at its corresponding points in
                `x`.
        """
        observations = [fun(xi) for fun, xi in zip(self.functions, x)]
        return torch.stack(observations)
