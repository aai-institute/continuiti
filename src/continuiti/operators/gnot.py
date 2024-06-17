"""
`continuiti.operators.gnot`

The GNOT (General Neural Operator Transformer) architecture.
"""

import torch
import torch.nn as nn
from torch.nn.functional import softmax

from continuiti.operators import Operator, OperatorShapes
from continuiti.networks.attention import MultiHead, HeterogeneousNormalized
from continuiti.networks import DeepResidualNetwork


class GNOTBlock(nn.Module):
    r"""GNOT block comprising the GNOT architecture.

    The GNOT block is the core element of the GNOT architecture. The block consists of multi-head cross attention,
    followed by residual expert networks, self attention, and a second block of expert networks. The expert networks
    are weighed after each expert block, using a gating mechanism. The gating mechanism is based on mixture-of-experts.
    The concept for this exact approach is described in Hao et al. The GNOT block is described in this paper as well.

    *Reference:* Hao, Z., Wang, Z., Su, H., Ying, C., Dong, Y., Liu, S., Cheng, Z., Song, J. and Zhu, J., 2023, July.
    Gnot: A general neural operator transformer for operator learning. In International Conference on Machine Learning
    (pp. 12556-12569). PMLR.

    Args:
        width: Embedding dimension and width of the deep residual networks.
        hidden_depth: Depth of each expert network.
        act: Activation function.
        n_heads: Number of attention heads.
        dropout_p: Temperature parameter governing the dropout rate.
        attention_class: Type of attention for both the cross and self attention.
        n_experts: the number of expert networks in each expert block.

    """

    def __init__(
        self,
        width: int,
        hidden_depth: int,
        act: nn.Module,
        n_heads: int,
        dropout_p: float,
        attention_class: type(nn.Module),
        n_experts: int,
    ):
        super().__init__()

        self.cross_attention = MultiHead(
            hidden_dim=width,
            n_heads=n_heads,
            attention=attention_class(),
            dropout_p=dropout_p,
            bias=True,
        )
        self.norm_cross_attention = nn.LayerNorm(width)

        self.ffn_1 = nn.ModuleList(
            [
                nn.Sequential(
                    DeepResidualNetwork(width, width, width, hidden_depth, act),
                    nn.LayerNorm(width),
                )
                for _ in range(n_experts)
            ]
        )

        self.self_attention = MultiHead(
            hidden_dim=width,
            n_heads=n_heads,
            attention=attention_class(),
            dropout_p=dropout_p,
            bias=True,
        )
        self.norm_self_attention = nn.LayerNorm(width)

        self.ffn_2 = nn.ModuleList(
            [
                nn.Sequential(
                    DeepResidualNetwork(width, width, width, hidden_depth, act),
                    nn.LayerNorm(width),
                )
                for _ in range(n_experts)
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch,
        attn_mask: torch.Tensor = None,
        gating_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        def gated_forward(
            src: torch.Tensor,
            ffn_module_list: nn.ModuleList,
            gating_mask_: torch.Tensor,
        ) -> torch.Tensor:
            if gating_mask_ is not None:
                res = torch.stack([expert(src) for expert in ffn_module_list], dim=0)
                res = res * gating_mask_
                res = torch.sum(res, dim=0)
            else:
                res = ffn_module_list[0](src)

            return res

        out = self.cross_attention(query, key, value, attn_mask=attn_mask) + query
        out = gated_forward(out, self.ffn_1, gating_mask) + out
        out = self.self_attention(out, out, out) + out
        return gated_forward(out, self.ffn_2, gating_mask) + out


class GNOT(Operator):
    r"""General Neural Operator Transformer (GNOT).

    The GNOT implementation uses GNOT blocks to compute the output of the operator. Each GNOT block consists of
    cross-attention, a mixture-of-experts block of neural networks, self attention, again followed by a
    mixture-of-experts neural network block.

    *Reference:* Hao, Z., Wang, Z., Su, H., Ying, C., Dong, Y., Liu, S., Cheng, Z., Song, J. and Zhu, J., 2023, July.
    Gnot: A general neural operator transformer for operator learning. In International Conference on Machine Learning
    (pp. 12556-12569). PMLR.

    Args:
        shapes: Shapes of the operator.
        encoding_depth: Depth of the encoding networks.
        width: Width of networks and embedding dim.
        hidden_depth: Depth of each hidden network.
        act: Activation function.
        n_blocks: Number of GNOT blocks in the operator.
        attention_class: Attention mechanism.
        n_heads: Number of attention heads.
        dropout_p: Temperature parameter controlling dropout probability.
        n_experts: Number of expert residual networks in every expert block.

    """

    def __init__(
        self,
        shapes: OperatorShapes,
        encoding_depth: int = 4,
        width: int = 32,
        hidden_depth=8,
        act: nn.Module = None,
        n_blocks: int = 1,
        attention_class: type(nn.Module) = HeterogeneousNormalized,
        n_heads: int = 1,
        dropout_p: float = 0.0,
        n_experts: int = 1,
    ):
        super().__init__()

        self.n_experts = n_experts

        if act is None:
            act = nn.GELU()

        self.query_encoder = DeepResidualNetwork(
            input_size=shapes.y.dim,
            width=width,
            output_size=width,
            depth=encoding_depth,
            act=act,
        )
        self.key_encoder = DeepResidualNetwork(
            input_size=shapes.x.dim + shapes.u.dim,
            width=width,
            output_size=width,
            depth=encoding_depth,
            act=act,
        )
        self.value_encoder = DeepResidualNetwork(
            input_size=shapes.x.dim + shapes.u.dim,
            width=width,
            output_size=width,
            depth=encoding_depth,
            act=act,
        )

        if n_experts > 1:
            self.expert_encoders = nn.ModuleList(
                [
                    DeepResidualNetwork(
                        input_size=shapes.y.dim,
                        width=width,
                        output_size=width,
                        depth=encoding_depth,
                        act=act,
                    )
                    for _ in range(n_experts)
                ]
            )
        self.blocks = nn.ModuleList(
            [
                GNOTBlock(
                    width=width,
                    hidden_depth=hidden_depth,
                    act=act,
                    n_heads=n_heads,
                    dropout_p=dropout_p,
                    attention_class=attention_class,
                    n_experts=n_experts,
                )
                for _ in range(n_blocks)
            ]
        )

        self.project = nn.Linear(width, shapes.v.dim)

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        y = y.transpose(1, 2)
        input_vec = torch.cat([x, u], dim=1).transpose(1, 2)

        gating_mask = None
        if self.n_experts > 1:
            gating_mask = torch.stack(
                [expert(y) for expert in self.expert_encoders], dim=0
            )
            gating_mask = softmax(gating_mask, dim=0)

        out = self.query_encoder(y)
        key = self.key_encoder(input_vec)
        value = self.value_encoder(input_vec)

        for block in self.blocks:
            out = block(
                query=out, key=key, value=value, attn_mask=None, gating_mask=gating_mask
            )

        return self.project(out).transpose(1, 2)
