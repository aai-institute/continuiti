import torch
import torch.nn as nn


class FunctionEncoder(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_head: int = 1,
        dropout_p: float = 0,
        bias: bool = True,
        depth: int = 1,
        act: nn.Module = None,
        ff_depth: int = 2,
    ):
        super().__init__()

        if act is None:
            act = nn.GELU()

        self.layers = nn.Sequential(
            *[
                FunctionEncoderLayer(
                    n_dim=n_dim,
                    n_head=n_head,
                    dropout_p=dropout_p,
                    bias=bias,
                    act=act,
                    ff_depth=ff_depth,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = src
        for mod in self.layers:
            out = mod(out)
        return out


class FunctionEncoderLayer(nn.Module):
    def __init__(
        self,
        n_dim: int,
        n_head: int = 1,
        dropout_p: float = 0,
        bias: bool = True,
        act: nn.Module = None,
        ff_depth: int = 2,
    ):
        super().__init__()

        if act is None:
            act = nn.GELU()

        # attention
        self.attn = nn.MultiheadAttention(n_dim, n_head, bias=bias)
        self.attn_norm = nn.LayerNorm(n_dim, eps=1e-5, bias=bias)
        self.attn_dropout = nn.Dropout()

        # feed forward
        forward_modules = [
            element
            for element in [
                nn.Linear(n_dim, n_dim, bias=bias),
                act,
                nn.Dropout(dropout_p),
            ]
            for _ in range(ff_depth - 1)
        ]
        forward_modules.append(
            nn.Linear(n_dim, n_dim, bias=bias)
        )  # last layer without activation function
        forward_modules.append(nn.Dropout(dropout_p))
        self.feed_forward = nn.Sequential(*forward_modules)
        self.feed_forward_norm = nn.LayerNorm(n_dim, eps=1e-5, bias=bias)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        out = src

        # self-attention block
        out = self.attn(out, out, out)[0]
        out = self.attn_dropout(out)
        out = out + src
        out = self.attn_norm(out)
        attn_out = out

        # feed forward block
        out = self.feed_forward(out)
        out = out + attn_out
        out = self.feed_forward_norm(out)

        return out
