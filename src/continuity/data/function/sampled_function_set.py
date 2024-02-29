"""
`continuity.data.function.sampled_function_set`

Sampled function set implementation.
"""

import torch

from .function_set import FunctionSet


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
