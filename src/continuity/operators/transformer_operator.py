import torch
import torch.nn as nn
from typing import Callable
from .operator import Operator
from .shape import OperatorShapes
from continuity.networks import DeepResidualNetwork, MultiHeadAttention, FunctionEncoder


class TransformerOperator(Operator):
    """Neural operator using transformer mechanisms for mapping input functions to output functions.

    The `TransformerOperator` leverages deep residual networks for input encoding, multi-head attention for capturing
    interactions between inputs, and a function encoder to further process the information within the function.

    Input Function Encodings
    Key Encoding: A deep residual network that encodes the input space x into the hidden dimension.
    Value Encoding: A deep residual network that encodes the function values u into the hidden dimension.
    Query Encoding: A deep residual network that encodes the output space y into the hidden dimension.

    Cross-Attention Mechanism
    Applies multi-head attention to the encoded query, key, and value representations to model the interactions between
    the input points and the function evaluations. Layer normalization and dropout are  applied to the output of the
    attention mechanism.

    Feed Forward Network
    A deep residual network processes the output of the cross-attention mechanism, followed by dropout and layer
    normalization for further processing and regularization.

    Function Encoder
    Block that further encodes the processed information, utilizing self-attention mechanism and feed-forward networks.

    Projection
    A linear projection layer that maps the final encoded representation to the output space v, producing the
    transformed function's values.

    Args:
         shapes: shape of the underlying mapping.
         hidden_dim: Dimension in which the information inside the transformer is processed (embedding dim).
         encoding_depth: Depth of deep residual networks producing the encodings for the cross attention.
         act: Activation function applied in deep residual networks and feed forward units.
         attention: Implementation of attentnion to use.
         n_heads: Number of attention heads used both in cross- and self-attention.
         bias: If True, adds bias to the projection layers of multi-head attention.
         dropout_p: Probability for dropout in dropout layers.
         feed_forward_depth: Depth of the deep residual network processing information after the initial
            cross-attention.
         function_encoder_depth: Number of FunctionEncoderLayer's in FunctionEncoder.
         function_encoder_layer_depth: Number of Layers in each FunctionEncoderLayer of FunctionEncoder.

    """

    def __init__(
        self,
        shapes: OperatorShapes,
        hidden_dim: int = 32,
        encoding_depth: int = 2,
        act: nn.Module = None,
        attention: Callable = nn.functional.scaled_dot_product_attention,
        n_heads: int = 1,
        bias: bool = True,
        dropout_p: float = 0.0,
        feed_forward_depth: int = 2,
        function_encoder_depth: int = 1,
        function_encoder_layer_depth: int = 2,
    ):
        super().__init__()
        self.shapes = shapes

        if act is None:
            act = nn.Tanh()

        # input function
        self.key_encoding = DeepResidualNetwork(
            input_size=shapes.x.dim,
            output_size=hidden_dim,
            width=hidden_dim,
            depth=encoding_depth,
            act=act,
        )
        self.val_encoding = DeepResidualNetwork(
            input_size=shapes.u.dim,
            output_size=hidden_dim,
            width=hidden_dim,
            depth=encoding_depth,
            act=act,
        )
        self.query_encoding = DeepResidualNetwork(
            input_size=shapes.y.dim,
            output_size=hidden_dim,
            width=hidden_dim,
            depth=encoding_depth,
            act=act,
        )
        self.cross_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            bias=bias,
            dropout_p=dropout_p,
            attention=attention,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim, eps=1e-5, bias=bias)
        self.cross_dropout = nn.Dropout(p=dropout_p)

        self.feed_forward = DeepResidualNetwork(
            input_size=hidden_dim,
            output_size=hidden_dim,
            width=hidden_dim,
            depth=feed_forward_depth,
            act=act,
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim, eps=1e-5, bias=bias)
        self.feed_forward_dropout = nn.Dropout(p=dropout_p)

        # function encoder blocks
        self.function_encoder = FunctionEncoder(
            n_dim=hidden_dim,
            depth=function_encoder_depth,
            n_head=n_heads,
            dropout_p=dropout_p,
            bias=bias,
            act=act,
            ff_depth=function_encoder_layer_depth,
        )

        # projection
        self.project = nn.Linear(hidden_dim, shapes.v.dim)

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # cross-attention block
        query = self.query_encoding(y)
        key = self.key_encoding(x)
        value = self.val_encoding(u)
        out = self.cross_attn(query, key, value)
        out = out + query
        out = self.cross_norm(out)
        out = self.cross_dropout(out)
        ca_out = out

        # feed forward
        out = self.feed_forward(out)
        out = self.feed_forward_dropout(out)
        out = ca_out + out
        out = self.feed_forward_norm(out)

        # function encoder
        out = self.function_encoder(out)

        # project into the output space
        return self.project(out)
