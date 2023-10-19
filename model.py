import numpy as np
import keras_core as keras
from data import Observation
from numpy import ndarray

class Model(keras.Model):
    """Base model mapping observations to evaluations."""

    def __init__(
            self,
            coordinate_dim: int,
            num_channels: int,
            num_sensors: int
        ):
        """A model maps observations to evaluations.
        
        Args:
            coordinate_dim: Dimension of coordinate space (d)
            num_channels: Number of channels (c)
            num_sensors: Number of input sensors (s)
        """
        super().__init__()

        # Dimension of coordinate space
        self.d = coordinate_dim

        # Number of channels
        self.c = num_channels

        # Number of sensors
        self.s = num_sensors


    def model(self, _):
        """Model implementation."""
        raise NotImplementedError


    def call(self, obs_pos: ndarray) -> ndarray:
        """Map batch of observations and positions to evaluations.
        
        Args:
            obs_pos: tensor of shape (batch_size, s * (d + c) + d)

        Returns:
            evals: tensor of shape (batch_size, c)
        """
        batch_size = obs_pos.shape[0]
        assert obs_pos.shape == (batch_size, 
                                 self.s * (self.d + self.c) + self.d)

        evals = self.model(obs_pos)
        assert evals.shape == (batch_size, self.c)

        return evals
    


class FNNModel(Model):
    """Fully connected neural network model."""

    def __init__(
            self,
            coordinate_dim: int,
            num_channels: int,
            num_sensors: int,
            width: int,
            depth: int,
        ):
        super().__init__(coordinate_dim, num_channels, num_sensors)

        self.input_layer = keras.layers.Dense(width, activation="tanh")
        self.hidden_layers = [
            keras.layers.Dense(width, activation="tanh")
            for _ in range(depth)
        ]
        self.output_layer = keras.layers.Dense(self.c)

    def model(self, x):
        x = self.input_layer(x)
            
        for layer in self.hidden_layers:
            x = layer(x) + x

        return self.output_layer(x)
