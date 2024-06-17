import torch
import torch.nn as nn
from torch.nn.functional import softmax

from continuiti.operators import Operator, OperatorShapes
from continuiti.networks.attention import MultiHead, HeterogeneousNormalized
from continuiti.networks import DeepResidualNetwork


class Encoder(nn.Module):
    def __init__(
        self,
        input_width: int,
        width: int,
        depth: int,
        act: nn.Module = None,
    ):
        super().__init__()

        if act is None:
            act = nn.GELU()

        self.model = nn.Sequential()

        self.model.add_module("lift_1", nn.Linear(input_width, width))
        self.model.add_module("norm_1", nn.LayerNorm(width))
        self.model.add_module("activation_1", act)

        for i in range(depth - 1):
            self.model.add_module(f"lift_{i + 2}", nn.Linear(width, width))
            self.model.add_module(f"norm_{i + 2}", nn.LayerNorm(width))
            self.model.add_module(f"activation_{i + 2}", act)

        self.model.add_module(f"lift_{depth + 1}", nn.Linear(width, width))
        self.model.add_module(f"norm_{depth + 1}", nn.LayerNorm(width))

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.model(src)


class GNOTBlock(nn.Module):
    def __init__(
        self,
        width: int,
        hidden_depth: int,
        act: nn.Module,
        n_heads: int,
        dropout_p: float,
        attention: type(nn.Module),
        n_experts: int,
    ):
        super().__init__()

        self.cross_attention = MultiHead(
            hidden_dim=width,
            n_heads=n_heads,
            attention=attention(),
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
            attention=attention(),
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
        out_ca = self.cross_attention(query, key, value, attn_mask=attn_mask) + query

        if gating_mask is not None:
            out = torch.stack([expert(out_ca) for expert in self.ffn_1], dim=0)
            out = out * gating_mask
            out = torch.sum(out, dim=0)
        else:
            out = self.ffn_1[0](out_ca)
        out = out + out_ca

        out_sa = self.self_attention(out, out, out) + out

        if gating_mask is not None:
            out = torch.stack([expert(out_sa) for expert in self.ffn_2], dim=0)
            out = out * gating_mask
            out = torch.sum(out, dim=0)
        else:
            out = self.ffn_2[0](out_sa)
        out = out + out_sa

        return out


class GNOT(Operator):
    def __init__(
        self,
        shapes: OperatorShapes,
        encoding_depth: int = 4,
        width: int = 32,
        hidden_depth=8,
        act: nn.Module = None,
        n_blocks: int = 1,
        attention: type(nn.Module) = HeterogeneousNormalized,
        n_heads: int = 4,
        dropout_p: float = 0.0,
        n_experts: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.n_experts = n_experts

        if act is None:
            act = nn.GELU()

        self.query_encoder = Encoder(
            width=width, input_width=shapes.y.dim, depth=encoding_depth
        )
        self.key_encoder = Encoder(
            width=width, input_width=shapes.x.dim + shapes.u.dim, depth=encoding_depth
        )
        self.value_encoder = Encoder(
            width=width, input_width=shapes.x.dim + shapes.u.dim, depth=encoding_depth
        )

        if n_experts > 1:
            self.expert_encoders = nn.ModuleList(
                [
                    Encoder(width=width, input_width=shapes.y.dim, depth=encoding_depth)
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
                    attention=attention,
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
