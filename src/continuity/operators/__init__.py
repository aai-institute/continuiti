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
from .neuraloperator import NeuralOperator
from .deeponet import DeepONet
from .belnet import BelNet
from .dno import DeepNeuralOperator
from .shape import OperatorShapes

__all__ = [
    "Operator",
    "OperatorShapes",
    "DeepONet",
    "NeuralOperator",
    "BelNet",
    "DeepNeuralOperator",
]
