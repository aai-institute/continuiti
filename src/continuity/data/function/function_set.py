"""
`continuity.data.function.function_set`

Function set.
"""

import torch
from typing import Callable, List, Union
from .function import Function


class FunctionSet:
    r"""Function set class.

    For a domain X supported on a field K1, and the codomain Y supported on a field K2, a function space is the set of
    functions F(X, Y) that map from X to Y and is closed with respect to addition
    $(f+g)(x):X \rightarrow Y, x\mapsto f(x)+g(x)$ and scalar multiplication $(cf)(x) \rightarrow Y, x\mapsto cf(x)$.
    In practice, we consider subsets of function spaces (where in general, these properties are not respected) and,
    therefore, this class implements a (parametrized) FunctionSet.

    Args:
        parameterized_mapping: A two level nested callable that takes a single parameter in the outer callable as
            argument and vectors x as inputs to the second callable.

    Example:
        ```python
        p_sine = FunctionSet(lambda a: lambda x: a * torch.sin(x))
        param = torch.arange(5)
        for func in sine(param):
            print(type(func))
        ```
        Out:
        ```shell
        <class 'continuity.data.function.function.Function'>
        <class 'continuity.data.function.function.Function'>
        <class 'continuity.data.function.function.Function'>
        <class 'continuity.data.function.function.Function'>
        <class 'continuity.data.function.function.Function'>
        ```

    """

    def __init__(self, parameterized_mapping: Callable):
        self.parameterized_mapping = parameterized_mapping

    def __call__(
        self, parameters: Union[torch.Tensor, List[torch.Tensor]]
    ) -> List[Function]:
        """Evaluates the function set for a specific discrete instance of parameters.

        Args:
            parameters: Parameters for which the mapping class argument will be evaluated of shape
                (n_observations, n_parameters) or list with n_observation elements with n_parameters each.

        Returns:
            List of Function instances for the given parameters of this function set instance.
        """
        funcs = []
        for param in parameters:

            def mapping(x, p=param):
                return self.parameterized_mapping(p)(x)

            funcs.append(Function(mapping))
        return funcs
