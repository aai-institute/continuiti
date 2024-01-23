"""
`continuity.operators`

Operators in Continuity.

Every operator maps collocation points `x`, function values `u`,
and evaluation points `y` to evaluations of `v`:

```
v = operator(x, u, y)
```
"""

from .operator import Operator
from .deeponet import DeepONet
from .neuraloperator import ContinuousConvolution, NeuralOperator

__all__ = ["Operator", "DeepONet", "ContinuousConvolution", "NeuralOperator"]
