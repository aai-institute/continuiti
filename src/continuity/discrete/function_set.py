import inspect
import torch
from typing import Callable, Any
from .function import Function


class FunctionSet(Function):
    r"""Function set class.

    For a domain X supported on a field K1, and the codomain Y supported on a field K2 a function space is the set of
    functions F(X, Y) that map from X to Y. With addition $(f+g)(x):X \rightarrow Y, x\mapsto f(x)+g(x)$ and scalar
    multiplication $(cf)(x) \rightarrow Y, x\mapsto cf(x)$ it becomes a vector space supported on functions F(x, Y). As
    these two properties can not always be respected (especially in the context of discrete representations of
    functions) we call this class FunctionSet.

    Args:
        mapping: A callable that takes exactly two parameters (parameters and location) with parameters first and
            location second.

    """

    def __init__(self, mapping: Callable):
        sig = inspect.signature(mapping)
        assert (
            len(sig.parameters) == 2
        ), "Mapping must have exactly two parameters (parameters and location)."

        super().__init__(mapping)

    def __call__(self, parameters: Any, locations: Any) -> torch.Tensor:
        """Evaluates the function set for a specific discrete instance of parameters and locations.

        Args:
            parameters: The parameters in which the function will be evaluated.
            locations: The location in which the function will be evaluated.

        Returns:
            function values for the given parameters and locations of this function set instance.
        """
        return super().__call__(parameters, locations)
