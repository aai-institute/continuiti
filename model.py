import torch
from time import time

device = torch.device("cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

print("Device:", device)


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

        # Move to device
        self.to(device)

        # Print number of model parameters
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {num_params}")
    

    def fit(self, dataset, epochs, writer=None):
        """Fit model to data set."""
        for epoch in range(epochs+1):
            mean_loss = 0

            start = time()
            for i in range(len(dataset)):
                u, v, x = dataset[i]
                self.optimizer.zero_grad()
                y_pred = self(u, x)
                loss = self.criterion(y_pred, v)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item()
            end = time()
            mean_loss /= len(dataset)

            if writer is not None:
                writer.add_scalar("Loss/train", mean_loss, epoch)

            iter_per_second = len(dataset) / (end - start)
            print(f"\rEpoch {epoch}:  loss = {mean_loss:.4e}  "
                  f"({iter_per_second:.2f} it/s)", end='')
        print("")


class FullyConnected(TorchModel):
    """Fully connected architecture."""

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

        self.input_size = num_sensors * (num_channels + coordinate_dim) \
            + coordinate_dim
        output_size = num_channels
        self.drn = DeepResidualNetwork(
            self.input_size,
            output_size,
            self.width,
            self.depth,
        )


    def forward(self, u, x):
        """Forward pass."""
        batch_size = u.shape[0]
        assert batch_size == x.shape[0]
        num_positions = x.shape[1]
        d = self.coordinate_dim

        ux = torch.empty(
            (batch_size, num_positions, self.input_size),
            device=device,
        )

        for i, u_tensor in enumerate(u):
            ux[i, :, :-d] = u_tensor.flatten()
        ux[:, :, -d:] = x

        ux = ux.reshape((batch_size * num_positions, -1))
        v = self.drn(ux)
        v = v.reshape((batch_size, num_positions, self.num_channels))
        return v


class DeepONet(TorchModel):
    """DeepONet architecture."""

    def __init__(
            self,
            coordinate_dim: int,
            num_channels: int,
            num_sensors: int,
            branch_width: int,
            branch_depth: int,
            trunk_width: int,
            trunk_depth: int,
            basis_functions: int,
        ):
        """A model maps observations to evaluations.
        
        Args:
            coordinate_dim: Dimension of coordinate space
            num_channels: Number of channels
            num_sensors: Number of input sensors
            branch_width: Width of branch network
            branch_depth: Depth of branch network
            trunk_width: Width of trunk network
            trunk_depth: Depth of trunk network
            basis_functions: Number of basis functions
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors
        self.basis_functions = basis_functions

        branch_input = num_sensors * (num_channels + coordinate_dim)
        trunk_input = coordinate_dim

        self.branch = DeepResidualNetwork(
            branch_input,
            self.num_channels * basis_functions,
            branch_width,
            branch_depth,
        )

        self.trunk = DeepResidualNetwork(
            trunk_input,
            self.num_channels * basis_functions,
            trunk_width,
            trunk_depth,
        )


    def forward(self, u, x):
        """Forward pass."""
        batch_size_u = u.shape[0]
        batch_size_x = x.shape[1]

        u = u.reshape((batch_size_u, -1))
        x = x.reshape((-1, self.coordinate_dim))

        b = self.branch(u)
        t = self.trunk(x)

        b = b.reshape((batch_size_u, self.basis_functions, self.num_channels))
        t = t.reshape((batch_size_u, batch_size_x, self.basis_functions, self.num_channels))

        sum = torch.einsum("ubc,uxbc->uxc", b, t)
        assert sum.shape == (batch_size_u, batch_size_x, self.num_channels)
        return sum


class ContinuousConvolutionLayer(torch.nn.Module):
    """Continuous convolution layer."""

    def __init__(
            self,
            coordinate_dim: int,
            num_channels: int,
            num_sensors: int,
            width: int,
            depth: int,
        ):
        """Maps continuous functions via convolution with a trainable kernel to another continuous functions using point-wise evaluations.
        
        Args:
            coordinate_dim: Dimension of coordinate space
            num_channels: Number of channels
            num_sensors: Number of input sensors
            width: Width of kernel network
            depth: Depth of kernel network
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors

        self.kernel = DeepResidualNetwork(1, 1, width, depth)


    def forward(self, yu, x):
        """Forward pass."""
        batch_size = yu.shape[0]
        num_sensors = yu.shape[1]
        assert batch_size == x.shape[0]
        x_size = x.shape[1]

        # Sensors are (x, u)
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
        assert integral.shape == (batch_size, x_size, self.num_channels)
        return integral



class NeuralOperator(TorchModel):
    """Neural operator architecture."""

    def __init__(
            self,
            coordinate_dim: int,
            num_channels: int,
            num_sensors: int,
            layers: int,
            kernel_width: int,
            kernel_depth: int,
        ):
        """Maps observations and positions to evaluations.
        
        Args:
            coordinate_dim: Dimension of coordinate space
            num_channels: Number of channels
            num_sensors: Number of input sensors
            layers: Number of layers
            width: Width of kernel network
            depth: Depth of kernel network
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors

        self.layers = torch.nn.ModuleList([
            ContinuousConvolutionLayer(
                coordinate_dim,
                num_channels,
                num_sensors,
                kernel_width,
                kernel_depth,
            ) for _ in range(layers)
        ])

        self.projection = ContinuousConvolutionLayer(
            coordinate_dim,
            num_channels,
            num_sensors,
            kernel_width,
            kernel_depth,
        )


    def forward(self, yu, x):
        """Forward pass."""
        batch_size = yu.shape[0]
        assert batch_size == x.shape[0]
        x_size = x.shape[1]

        sensor_dim = self.coordinate_dim + self.num_channels
        assert yu.shape == (batch_size, self.num_sensors, sensor_dim)

        # First input is u
        yv = yu

        # Sensors positions are equal across all layers
        y = yv[:, :, -self.coordinate_dim:]

        # Initial value
        v = yv[:, :, :-self.coordinate_dim]

        # Layers
        for layer in self.layers:
            # Layer operation (with residual connection)
            v = layer(yv, y) + v
            assert v.shape == (batch_size, self.num_sensors, self.num_channels)

            # Activation
            v = torch.tanh(v)

            yv = torch.cat((y, v), axis=-1)
            assert yv.shape == (batch_size, self.num_sensors, sensor_dim)

        # Projection layer
        w = self.projection(yv, x)
        assert w.shape == (batch_size, x_size, self.num_channels)
        return w

