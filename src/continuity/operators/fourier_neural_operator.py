# %%

'''

TODO:
    - Think about how to define shapes properly, FFT needs uniform grid -> Reshaping
    - generalize to N dimensions
    - Generalize kernel to (x-dim, y-dim)
    - Add W and bias: Why is this a convolution in the repo of the original paper?
    - x and y are ignored! Should we add options to make kernel dependent?
    - Think about how to make Operator more layer like

    
    K(u) + W * u(x) + b
FourierLayer
'''

from continuity.operators import Operator
import torch
from continuity.data import DatasetShapes
import torch.nn as nn
import math
from torch.fft import rfft, irfft, rfftn, irfftn

class FNO1d(Operator):

    def __init__(
        self,
        shapes: DatasetShapes,
        num_frequencies: int | None = None,
    ) -> None:
        super().__init__()

        assert shapes.x.dim == 1

        self.num_frequencies = shapes.x.num // 2 + 1 if num_frequencies is None else num_frequencies
        assert self.num_frequencies > shapes.x.num // 2,\
            "num_frequencies is too large. The fft of a real valued function has only (shapes.x.num // 2 + 1) unique frequencies."

        assert shapes.u.dim == shapes.v.dim

        shape = (self.num_frequencies, shapes.u.dim, shapes.u.dim)
        weigths_real = torch.Tensor(*shape)
        weigths_img = torch.Tensor(*shape)

        # initialize
        nn.init.kaiming_uniform_(weigths_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(weigths_img, a=math.sqrt(5))

        self.weights_complex = torch.complex(weigths_real, weigths_img)
        self.kernel = torch.nn.Parameter(self.weights_complex)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:

        #assert u.shape[:-1] == x.shape[:-1] == y.shape[:-1]
        del x, y # not used

        # f(w) = f*(-w)
        u_fourier = rfft(u, axis=1) # real-valued fft -> skip negative frequencies
        out_fourier = torch.einsum('nds,bns->bnd', self.kernel, u_fourier[:, :self.num_frequencies, :])
        out = irfft(out_fourier, axis=1, n=u.shape[1])

        return out


# FourierLayer
# 
class FNO(Operator):

    def __init__(
        self,
        shapes: DatasetShapes,
        num_frequencies: int | None = None,
    ) -> None:
        super().__init__()

        # u.shape = (batchsize, x.num, u.dim) -> (100, 16, 2)
        # u.shape = (100, 4, 4)

        # is this still correct?
        points_per_dimension = int(shapes.x.num ** (1 / shapes.x.dim))
        self.num_frequencies = points_per_dimension // 2 + 1 if num_frequencies is None else num_frequencies
        assert self.num_frequencies > points_per_dimension // 2,\
            "num_frequencies is too large. The fft of a real valued function has only (shapes.x.num // 2 + 1) unique frequencies."

        assert shapes.u.dim == shapes.v.dim

        self.shapes = shapes

        shape = (self.num_frequencies,) * self.shapes.x.dim + (shapes.u.dim, shapes.u.dim)
        weigths_real = torch.Tensor(*shape)
        weigths_img = torch.Tensor(*shape)

        # initialize
        nn.init.kaiming_uniform_(weigths_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(weigths_img, a=math.sqrt(5))

        self.weights_complex = torch.complex(weigths_real, weigths_img)
        self.kernel = torch.nn.Parameter(self.weights_complex)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:

        #assert u.shape[:-1] == x.shape[:-1] == y.shape[:-1]
        del x, y # not used

        batch_size = u.shape[0]
        u = self._reshape(u)

        print(f"u-shape {u.shape}")
        print(f"kernel-shape {self.kernel.shape}")

        u_fourier = rfftn(u, dim=list(range(1, u.dim()-1))) # real-valued fft -> skip negative frequencies

        frequency_indices = "".join("acefghijklmnopqrtuvxyz"[:int(u.dim()-2)])
        contraction_string = "{}ds,b{}s->b{}d".format(frequency_indices, frequency_indices, frequency_indices)

        print(f"contraction string: {contraction_string}")


        slices = [slice(batch_size)] + [slice(self.num_frequencies)]*(u.dim() - 2) + [slice(u.shape[-1])]
        u_fourier_sliced = u_fourier[slices]

        print(f"slices  {u_fourier_sliced.shape}")

        out_fourier = torch.einsum(contraction_string, self.kernel, u_fourier_sliced)
        print(f"Output fourier after tensor {out_fourier.shape}")
        print(f"dim={list(range(1, u.dim()))}, s={u.shape[1:-1]}")
        out = irfftn(out_fourier, dim=list(range(1, u.dim()-1)), s=u.shape[1:-1]) # TODO: u.shape will change

        # reshape
        out = out.reshape(batch_size, -1, self.shapes.u.dim)

        return out


    def _reshape(
        self,
        u,
    ):
        batch_size = u.shape[0]
        points_per_dim = int((self.shapes.x.num)**(1./self.shapes.x.dim))

        # TODO: check for consitency, add assert 

        shape = [batch_size] + [points_per_dim for _ in range(self.shapes.x.dim)] + [self.shapes.u.dim]
        u = u.reshape(shape)
        return u


class FNO1d_model(Operator):

    def __init__(self, shapes: DatasetShapes, num_layers: int = 5):

        super().__init__()

        self.lifting = lambda x: x
        self.projection = lambda x: x

        self.num_layers = num_layers
        self.act = nn.ReLU()

        self.layers = torch.nn.ModuleList([FNO1d(shapes) for _ in range(num_layers)])

    def forward(self, x, u, y):

        u = self.lifting(u)

        for layer in self.layers[:-1]:
            u = layer(x, u, y)
            u = self.act(u)
        
        u = self.layers[-1](x, u, y) # apply last layer without activation
        u = self.projection(u)

        return u


# %%
