"""
`continuity.operators.fno`

The Fourier Neural Operator
"""

import torch
from typing import Optional
from continuity.operators import NeuralOperator
from continuity.operators.fourierlayer import FourierLayer
from continuity.operators.shape import OperatorShapes, TensorShape


class FourierNeuralOperator(NeuralOperator):
    r"""Fourier Neural Operator (FNO) architecture

    *Reference:* Z. Li et al. Fourier Neural Operator for Parametric Partial
      Differential Equations arXiv:2010.08895 (2020)

    Args:
        shapes: Shapes of the input and output data.
        depth: Number of Fourier layers.
        width: Latent dimension of the Fourier layers.
        act: Activation function. Default is tanh.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        depth: int = 3,
        width: int = 3,
        act: Optional[torch.nn.Module] = None,
    ):
        self.depth = depth
        self.width = width
        self.act = act

        latent_shapes = OperatorShapes(
            x=shapes.x,
            u=TensorShape(shapes.u.num, width),
            y=shapes.x,
            v=TensorShape(shapes.u.num, width),
        )

        layers = torch.nn.ModuleList(
            [FourierLayer(latent_shapes) for _ in range(depth)]
        )

        super().__init__(shapes, layers, act)

    def __str__(self):
        """String representation of the operator."""
        s = "FourierNeuralOperator("
        s += f"depth={self.depth}, width={self.width}"
        if not isinstance(self.act, torch.nn.Tanh):
            s += f", {self.act}"
        s += ")"
        return s
