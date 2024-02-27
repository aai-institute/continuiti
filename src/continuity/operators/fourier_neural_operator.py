# %%

'''

TODO:
    - Think about how to define shapes properly, FFT needs uniform grid -> Reshaping
    - generalize to N dimensions
    - Add W and bias: Why is this a convolution in the repo of the original paper?
    - x and y are ignored! Should we add options to make kernel dependent?
    - Think about how to make Operator more layer like

'''

from continuity.operators import Operator
import torch
from continuity.data import DatasetShapes
import torch.nn as nn
import math
from torch.fft import rfft, irfft

class FNO1d(Operator):

    def __init__(
        self,
        shapes: DatasetShapes,
        num_frequencies: int | None = None,
    ) -> None:
        super().__init__()

        self.num_frequencies = shapes.x.num // 2 + 1 if num_frequencies is None else num_frequencies
        assert self.num_frequencies > shapes.x.num // 2,\
            "num_frequencies is too large. The fft of a real valued function has only (shapes.x.num // 2 + 1) unique frequencies."

        assert shapes.u.dim == shapes.v.dim

        shape = (self.num_frequencies, shapes.x.dim, shapes.x.dim)
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

        u_fourier = rfft(u, axis=1) # real-valued fft -> skip negative frequencies
        out_fourier = torch.einsum('nds,bns->bnd', self.kernel, u_fourier[:, :self.num_frequencies, :])
        out = irfft(out_fourier, axis=1, n=u.shape[1])

        return out



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

