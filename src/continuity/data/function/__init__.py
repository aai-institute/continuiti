"""
`continuity.data.function`

Functions and function-sets in Continuity.
"""

from .function import Function
from .parameterized_function import ParameterizedFunction
from .function_set import FunctionSet
from .sampled_function_set import SampledFunctionSet

__all__ = [
    "Function",
    "ParameterizedFunction",
    "FunctionSet",
    "SampledFunctionSet",
]
