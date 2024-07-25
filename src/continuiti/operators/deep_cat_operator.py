"""
`continuiti.operators.deep_cat_operator`

The DeepCatOperator architecture.
"""

import torch
import torch.nn as nn
from typing import Optional
from math import ceil, prod
from .operator import Operator, OperatorShapes
from continuiti.networks import DeepResidualNetwork


class DeepCatOperator(Operator):
    """Deep Cat Operator.

    This class implements the DeepCatOperator, a neural operator inspired by the DeepONet. It consists of three main
    parts:
    1. **Input Network**: Analogous to the "branch network," it processes the sensor inputs (`u`).
    2. **Eval Network**: Analogous to the "trunk network," it processes the evaluation locations (`y`).
    3. **Cat Network**: Combines the outputs from the Input and Eval Networks to produce the final output.

    The architecture offers three potential advantages:
    1. It allows the operator to integrate evaluation locations earlier, enabling a higher level of adaptive
        abstraction.
    2. The hyperparameters can be thought of as a control mechanism, dictating the flow of information. The
        `input_cat_ratio` hyperparameter provides a control mechanism for the information flow, allowing fine-tuning of
        the contributions from the Input and Eval Networks.
    3. It can achieve a high level of abstraction without relying on learning basis functions, evaluated in a single
        operation (dot product).

    Args:
        shapes: Operator shapes.
        input_net_width: Width of the input net (deep residual network). Defaults to 32.
        input_net_depth: Depth of the input net (deep residual network). Defaults to 4.
        eval_net_width: Width of the eval net (deep residual network). Defaults to 32.
        eval_net_depth: Depth of the eval net (deep residual network). Defaults to 4.
        input_cat_ratio: Ratio indicating how many values of the concatenated tensor originates from the input net.
            Controls flow of information into input- and eval-net. Defaults to 0.5.
        cat_net_width: Width of the cat net (deep residual network). Defaults to 32.
        cat_net_depth: Depth of the cat net (deep residual network). Defaults to 4.
        act: Activation function. Defaults to Tanh.
        device: Device.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        input_net_width: int = 32,
        input_net_depth: int = 4,
        eval_net_width: int = 32,
        eval_net_depth: int = 4,
        input_cat_ratio: float = 0.5,
        cat_net_width: int = 32,
        cat_net_depth: int = 4,
        act: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(shapes=shapes, device=device)

        if act is None:
            act = nn.Tanh()

        assert (
            1.0 > input_cat_ratio > 0.0
        ), f"Ratio has to be in [0, 1], but found {input_cat_ratio}"
        input_out_width = ceil(cat_net_width * input_cat_ratio)
        assert (
            input_out_width != cat_net_width
        ), f"Input cat ratio {input_cat_ratio} results in eval net width equal zero."

        input_in_width = prod(shapes.u.size) * shapes.u.dim
        self.input_net = DeepResidualNetwork(
            input_size=input_in_width,
            output_size=input_out_width,
            width=input_net_width,
            depth=input_net_depth,
            act=act,
            device=device,
        )

        eval_out_width = cat_net_width - input_out_width
        self.eval_net = DeepResidualNetwork(
            input_size=shapes.y.dim,
            output_size=eval_out_width,
            width=eval_net_width,
            depth=eval_net_depth,
            act=act,
            device=device,
        )

        self.cat_act = act  # no activation before first and after last layer

        self.cat_net = DeepResidualNetwork(
            input_size=cat_net_width,
            output_size=shapes.v.dim,
            width=cat_net_width,
            depth=cat_net_depth,
            act=act,
            device=device,
        )

    def forward(
        self, _: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            _: Tensor containing sensor locations. Ignored.
            u: Tensor containing values of sensors. Of shape (batch_size, u_dim, num_sensors...).
            y: Tensor containing evaluation locations. Of shape (batch_size, y_dim, num_evaluations...).

        Returns:
            Tensor of predicted evaluation values. Of shape (batch_size, v_dim, num_evaluations...).
        """
        ipt = torch.flatten(u, start_dim=1)
        ipt = self.input_net(ipt)

        y_num = y.shape[2:]
        eval = y.flatten(start_dim=2).transpose(1, -1)
        eval = self.eval_net(eval)

        ipt = ipt.unsqueeze(1).expand(-1, eval.size(1), -1)
        cat = torch.cat([ipt, eval], dim=-1)
        out = self.cat_act(cat)
        out = self.cat_net(out)

        return out.reshape(-1, self.shapes.v.dim, *y_num)
