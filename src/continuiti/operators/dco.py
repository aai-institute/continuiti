"""
`continuiti.operators.dco`

The DeepCatOperator (DCO) architecture.
"""

import torch
import torch.nn as nn
from typing import Optional
from math import ceil, prod
from .operator import Operator, OperatorShapes
from continuiti.networks import DeepResidualNetwork


class DeepCatOperator(Operator):
    """Deep Cat Operator.

    This class implements the DeepCatOperator, a neural operator inspired by the DeepONet.

    It consists of three main parts:

    1. **Branch Network**: Processes the sensor inputs (`u`).
    2. **Trunk Network**: Processes the evaluation locations (`y`).
    3. **Cat Network**: Combines the outputs from the Branch- and Trunk-Network to produce the final output.

    The architecture has the following structure:

    ````
    ┌─────────────────────┐    ┌────────────────────┐
    │ *Branch Network*    │    │ *Trunk Network*    │
    │  Input (u)          │    │ Input (y)          │
    │  Output (b)         │    │ Output (t)         │
    └─────────────────┬───┘    └──┬─────────────────┘
    ┌ ─ ─ ─ ─ ─ ─ ─ ─ ┴ ─ ─ ─ ─ ─ ┴ ─ ─ ─ ─ ─ ─ ─ ─ ┐
    │ *Concatenation*                               │
    │ Input (b, t)                                  │
    │ Output (c)                                    │
    │ branch_cat_ratio = b.numel() / cat_net_width  │
    └ ─ ─ ─ ─ ─ ─ ─ ─ ┴ ─ ─ ─ ─ ┴ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                  ┌─────────┴────────┐
                  │ *Cat Network*    │
                  │  Input (c)       │
                  │  Output (v)      │
                  └──────────────────┘
    ````

    This allows the operator to integrate evaluation locations earlier, while ensuring that both the sensor inputs and
    the evaluation location contribute in a predictable form to the flow of information. Directly stacking both the
    sensors and evaluation location can lead to an imbalance in the number of features in the neural operator. The
    arg `branch_cat_ratio` dictates how this fraction is set (defaults to 50/50). The cat-network does not require the
    neural operator to learn good basis functions with the trunk network only. The information from the input space and
    the evaluation locations can be taken into account early, allowing for better abstraction.

    Args:
        shapes: Operator shapes.
        branch_width: Width of the branch net (deep residual network). Defaults to 32.
        branch_depth: Depth of the branch net (deep residual network). Defaults to 4.
        trunk_width: Width of the trunk net (deep residual network). Defaults to 32.
        trunk_depth: Depth of the trunk net (deep residual network). Defaults to 4.
        branch_cat_ratio: Ratio indicating which fraction of the concatenated tensor originates from the branch net.
            Controls flow of information into branch- and trunk-net. Defaults to 0.5.
        cat_net_width: Width of the cat net (deep residual network). Defaults to 32.
        cat_net_depth: Depth of the cat net (deep residual network). Defaults to 4.
        act: Activation function. Defaults to Tanh.
        device: Device.

    """

    def __init__(
        self,
        shapes: OperatorShapes,
        branch_width: int = 32,
        branch_depth: int = 4,
        trunk_width: int = 32,
        trunk_depth: int = 4,
        branch_cat_ratio: float = 0.5,
        cat_net_width: int = 32,
        cat_net_depth: int = 4,
        act: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(shapes=shapes, device=device)

        if act is None:
            act = nn.Tanh()

        assert (
            0.0 < branch_cat_ratio < 1.0
        ), f"Ratio has to be in (0, 1), but found {branch_cat_ratio}"
        branch_out_width = ceil(cat_net_width * branch_cat_ratio)
        assert (
            branch_out_width != cat_net_width
        ), f"Input cat ratio {branch_cat_ratio} results in eval net width equal zero."

        input_in_width = prod(shapes.u.size) * shapes.u.dim
        self.branch_net = DeepResidualNetwork(
            input_size=input_in_width,
            output_size=branch_out_width,
            width=branch_width,
            depth=branch_depth,
            act=act,
            device=device,
        )

        eval_out_width = cat_net_width - branch_out_width
        self.trunk_net = DeepResidualNetwork(
            input_size=shapes.y.dim,
            output_size=eval_out_width,
            width=trunk_width,
            depth=trunk_depth,
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
            u: Tensor containing values of sensors of shape (batch_size, u_dim, num_sensors...).
            y: Tensor containing evaluation locations of shape (batch_size, y_dim, num_evaluations...).

        Returns:
            Tensor of predicted evaluation values of shape (batch_size, v_dim, num_evaluations...).
        """
        ipt = torch.flatten(u, start_dim=1)
        ipt = self.branch_net(ipt)

        y_num = y.shape[2:]
        eval = y.flatten(start_dim=2).transpose(1, -1)
        eval = self.trunk_net(eval)

        ipt = ipt.unsqueeze(1).expand(-1, eval.size(1), -1)
        cat = torch.cat([ipt, eval], dim=-1)
        out = self.cat_act(cat)
        out = self.cat_net(out)

        return out.reshape(-1, self.shapes.v.dim, *y_num)
