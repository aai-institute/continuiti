"""
`continuity.operators`

Operators in Continuity.

Every operator maps collocation points `x`, function values `u`,
and evaluation points `y` to evaluations of `v`:

```
v = operator(x, u, y)
```
"""

from .common import DeepResidualNetwork
from .operator import Operator
from .neuraloperator import NeuralOperator
from .deeponet import DeepONet
from .belnet import BelNet

__all__ = [
    "Operator",
    "DeepONet",
    "NeuralOperator",
    "DeepResidualNetwork",
    "BelNet",
]
