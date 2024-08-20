"""
**continuiti** is a Python package for machine learning on function operators.

The package is structured into the following modules:

::cards:: cols=2

- title: Benchmarks
  content: Benchmarks for testing operator architectures.
  url: benchmarks/index.md

- title: Data
  content: Data sets and data utility functions.
  url: data/index.md

- title: Discrete
  content: Discretization utilities like samplers.
  url: discrete/index.md

- title: Networks
  content: Neural network implementations.
  url: networks/index.md

- title: Operators
  content: Neural operator implementations.
  url: operators/index.md

- title: PDE
  content: Loss functions for physics-informed training.
  url: pde/index.md

- title: Trainer
  content: Default training loop for operator models.
  url: trainer/index.md

- title: Transforms
  content: Transformations for operator inputs/outputs.
  url: transforms/index.md

::/cards::

"""

__all__ = [
    "benchmarks",
    "data",
    "discrete",
    "networks",
    "operators",
    "pde",
    "trainer",
    "transforms",
]

from . import benchmarks
from . import data
from . import discrete
from . import networks
from . import operators
from . import pde
from . import trainer
from . import transforms

from .trainer import Trainer  # noqa
