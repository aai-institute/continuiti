import keras_core as keras
from numpy import ndarray


class ResidualBlock(keras.layers.Layer):
    """Fully connected residual block layer."""
    def __init__(self, width: int):
        super().__init__()

        self.input_layer = keras.layers.Input((width,))
        self.inner_layer = keras.layers.Dense(width)
        self.activation = keras.layers.Activation("tanh")
        self.outer_layer = keras.layers.Dense(width)

    def call(self, x):
        f = self.inner_layer(x)
        f = self.activation(f)
        f = self.outer_layer(f)
        return f + x


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

        input_size = coordinate_dim + num_sensors * num_channels

        self.input_layer = keras.layers.Input((input_size,))
        self.first_layer = keras.layers.Dense(width)
        self.hidden_layers = [ResidualBlock(width) for _ in range(depth)]
        self.last_layer = keras.layers.Dense(num_channels)


    def call(self, x: ndarray) -> ndarray:
        """Map batch of observations and positions to evaluations.
        
        Args:
            obs_pos: batch of flattened observation and position tensors

        Returns:
            evaluation at position
        """
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_layer(x)