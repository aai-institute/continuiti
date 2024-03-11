"""
`continuity.data.function_dataset.function_set`

Function set.
"""

import torch
from typing import Callable, List
from .function import Function


class FunctionSet(Function):
    r"""Function set class.

    For a domain X supported on a field K1, and the codomain Y supported on a field K2 a function space is the set of
    functions F(X, Y) that map from X to Y. With addition $(f+g)(x):X \rightarrow Y, x\mapsto f(x)+g(x)$ and scalar
    multiplication $(cf)(x) \rightarrow Y, x\mapsto cf(x)$ it becomes a vector space supported on functions F(x, Y). As
    these two properties can not always be respected (especially in the context of discrete representations of
    functions) we call this class FunctionSet.

    Args:
        mapping: A two level nested callable that takes a single parameter in the outer callable as argument and vectors
            x as inputs to the second callable.

    Example:
        ```python
        sine = FunctionSet(lambda a: lambda x: a * torch.sin(x))
        param = torch.arange(5)
        print(len(sine(param)))
        ```
        Out:
        ```shell
        5
        ```

    """

    def __init__(self, mapping: Callable):
        super().__init__(mapping)

    def __call__(self, parameters: torch.Tensor) -> List[Function]:
        """Evaluates the function set for a specific discrete instance of parameters and locations.

        Args:
            parameters: The parameters in which the function will be evaluated.

        Returns:
            function values for the given parameters and locations of this function set instance.
        """
        funcs = []
        for param in parameters:
            funcs.append(self.mapping(param))
        return funcs
