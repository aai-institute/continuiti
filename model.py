import keras_core as keras
from numpy import ndarray


class ResidualLayer(keras.layers.Layer):
    """Fully connected residual layer."""
    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def build(self, input_shape=None):
        assert input_shape is None or input_shape[-1] == self.width
        self.layer = keras.layers.Dense(self.width, activation="tanh")

    def call(self, x):
        return self.layer(x) + x


class DNN(keras.Model):
    """Fully connected deep neural network."""
    def __init__(
            self,
            input_size: int,
            output_size: int,
            width: int,
            depth: int,
        ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.width = width
        self.depth = depth

    def build(self, input_shape=None):
        self.input_layer = keras.layers.Input(input_shape)
        self.first_layer = keras.layers.Dense(self.width)
        self.hidden_layers = [
            ResidualLayer(self.width) for _ in range(self.depth)
        ]
        self.last_layer = keras.layers.Dense(self.output_size)

    def call(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_layer(x)


class ContinuityModel(keras.Model):
    """Continuity model mapping observations to evaluation."""

    def __init__(
            self,
            coordinate_dim: int,
            num_channels: int,
            num_sensors: int,
            width: int,
            depth: int,
        ):
        """A model maps observations to evaluations.
        
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

        self.input_shape = (None, coordinate_dim + num_sensors * num_channels)

    def build(self, input_shape=None):
        assert input_shape is None or input_shape[1] == self.input_shape[1]
        self.dnn = DNN(
            self.input_shape,
            self.num_channels,
            self.width,
            self.depth,
        )

    def call(self, x: ndarray) -> ndarray:
        """Map batch of observations and positions to evaluations.
        
        Args:
            obs_pos: batch of flattened observation and position tensors

        Returns:
            evaluation at position
        """
        return self.dnn(x)
