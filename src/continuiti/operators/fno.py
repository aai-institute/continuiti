"""
`continuiti.operators.fno`

The Fourier Neural Operator
"""

import torch
from typing import Optional
from continuiti.operators import NeuralOperator
from continuiti.operators.fourierlayer import FourierLayer
from continuiti.operators.shape import OperatorShapes, TensorShape


class FourierNeuralOperator(NeuralOperator):
    r"""Fourier Neural Operator (FNO) architecture

    *Reference:* Z. Li et al. Fourier Neural Operator for Parametric Partial
      Differential Equations arXiv:2010.08895 (2020)

    Args:
        shapes: Shapes of the input and output data.
        depth: Number of Fourier layers.
        width: Latent dimension of the Fourier layers.
        act: Activation function.
        device: Device.
        **kwargs: Additional arguments for the Fourier layers.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        depth: int = 3,
        width: int = 3,
        act: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        latent_shapes = OperatorShapes(
            x=shapes.x,
            u=TensorShape(width, shapes.u.size),
            y=shapes.x,
            v=TensorShape(width, shapes.u.size),
        )
        output_shapes = OperatorShapes(
            x=shapes.x,
            u=TensorShape(width, shapes.u.size),
            y=shapes.y,
            v=TensorShape(width, shapes.v.size),
        )

        layers = []
        for _ in range(depth - 1):
            layers += [FourierLayer(latent_shapes, device=device, **kwargs)]
        layers += [FourierLayer(output_shapes, device=device, **kwargs)]

        layers = torch.nn.ModuleList(layers)

        super().__init__(shapes, layers, act, device)
