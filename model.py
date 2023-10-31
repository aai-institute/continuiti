import torch


class ResidualLayer(torch.nn.Module):
    """Residual layer."""
    def __init__(self, width: int):
        super().__init__()
        self.layer = torch.nn.Linear(width, width)
        self.act = torch.nn.Tanh()

    def forward(self, x):
        return self.act(self.layer(x)) + x


class DeepResidualNetwork(torch.nn.Module):
    """Deep residual network."""
    def __init__(
            self,
            input_size: int,
            output_size: int,
            width: int,
            depth: int,
        ):
        super().__init__()

        self.first_layer = torch.nn.Linear(input_size, width)
        self.hidden_layers = torch.nn.ModuleList([
            ResidualLayer(width) for _ in range(depth)
        ])
        self.last_layer = torch.nn.Linear(width, output_size)

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_layer(x)


class TorchModel(torch.nn.Module):
    """Torch model."""
    
    def compile(self, optimizer, criterion):
        """Compile model."""
        self.optimizer = optimizer
        self.criterion = criterion
    

    def fit(self, dataset, epochs):
        """Fit model to data set."""
        for epoch in range(epochs+1):
            mean_loss = 0
            for i in range(len(dataset)):
                u, v, x = dataset[i]
                self.optimizer.zero_grad()
                y_pred = self(u, x)
                loss = self.criterion(y_pred, v)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item()

            mean_loss /= len(dataset)
            print(f'\rEpoch {epoch}:  loss = {mean_loss:.4e}', end='')
        print("")


class FullyConnectedModel(TorchModel):
    """Fully connected model."""

    def __init__(
            self,
            coordinate_dim: int,
            num_channels: int,
            num_sensors: int,
            width: int,
            depth: int,
        ):
        """Maps observations and positions to evaluations.
        
        Args:
            coordinate_dim: Dimension of coordinate space
            num_channels: Number of channels
            num_sensors: Number of input sensors
            width: Width of network
            depth: Depth of network
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors
        self.width = width
        self.depth = depth

        input_size = num_sensors * (num_channels + coordinate_dim) \
            + coordinate_dim
        output_size = num_channels
        self.drn = DeepResidualNetwork(
            input_size,
            output_size,
            self.width,
            self.depth,
        )


    def forward(self, u, x):
        """Forward pass."""
        batch_size = u.shape[0]
        assert batch_size == x.shape[0]
        num_sensors = x.shape[1]

        v = torch.zeros((batch_size, num_sensors, self.num_channels))
        for j in range(num_sensors):
            ux = torch.stack([
                torch.cat([u[i].flatten(), x[i, j]])
                for i in range(batch_size)
            ])
            v[:, j] = self.drn(ux)
        return v


# class DeepONet(TorchModel):
#     """DeepONet architecture."""

#     def __init__(
#             self,
#             coordinate_dim: int,
#             num_channels: int,
#             num_sensors: int,
#             branch_width: int,
#             branch_depth: int,
#             trunk_width: int,
#             trunk_depth: int,
#             basis_functions: int,
#         ):
#         """A model maps observations to evaluations.
        
#         Args:
#             coordinate_dim: Dimension of coordinate space
#             num_channels: Number of channels
#             num_sensors: Number of input sensors
#             branch_width: Width of branch network
#             branch_depth: Depth of branch network
#             trunk_width: Width of trunk network
#             trunk_depth: Depth of trunk network
#             basis_functions: Number of basis functions
#         """
#         self.coordinate_dim = coordinate_dim
#         self.num_channels = num_channels
#         self.num_sensors = num_sensors

#         branch_input = num_sensors * num_channels
#         trunk_input = coordinate_dim

#         self.inputs = keras.Input(shape=(branch_input + trunk_input,))

#         self.branch = DNN(
#             (None, branch_input),
#             self.num_channels * basis_functions,
#             branch_width,
#             branch_depth,
#         )(self.inputs[:, :branch_input])

#         self.trunk = DNN(
#             (None, trunk_input),
#             self.num_channels * basis_functions,
#             trunk_width,
#             trunk_depth,
#         )(self.inputs[:, -trunk_input:])

#         self.outputs = keras.ops.einsum("ij,ik->i", self.branch, self.trunk)
#         self.outputs = keras.layers.Reshape((self.num_channels,))(self.outputs)

#         super().__init__(inputs=self.inputs, outputs=self.outputs)


class NeuralOperator(TorchModel):
    """Neural operator architecture."""

    def __init__(
            self,
            coordinate_dim: int,
            num_channels: int,
            num_sensors: int,
            width: int,
            depth: int,
        ):
        """Maps observations and positions to evaluations.
        
        Args:
            coordinate_dim: Dimension of coordinate space
            num_channels: Number of channels
            num_sensors: Number of input sensors
            width: Width of kernel network
            depth: Depth of kernel network
        """
        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors

        super().__init__()
        self.kernel = DeepResidualNetwork(1, 1, width, depth)


    def forward(self, yu, x):
        """Forward pass."""
        batch_size = yu.shape[0]
        num_sensors = yu.shape[1]
        assert batch_size == x.shape[0]
        x_size = x.shape[1]
        
        # Sensors
        y = yu[:, :, -self.coordinate_dim:]
        u = yu[:, :, :-self.coordinate_dim]
        assert y.shape == (batch_size, num_sensors, self.coordinate_dim) 
        assert u.shape == (batch_size, num_sensors, self.num_channels) 

        # Compute radial coordinates
        assert x.shape == (batch_size, x_size, self.coordinate_dim)
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        r = torch.sum((x - y)**2, axis=-1)
        assert r.shape == (batch_size, num_sensors, x_size)

        # Flatten to 1D
        r = r.reshape((-1, 1))

        # Kernel operation
        k = self.kernel(r)

        # Reshape to 3D
        k = k.reshape((batch_size, num_sensors, x_size))

        # Compute integral
        integral = torch.einsum("bsx,bsc->bxc", k, u) / num_sensors

        return integral

