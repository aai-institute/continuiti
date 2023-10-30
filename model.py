import keras_core as keras


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
        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors
        self.width = width
        self.depth = depth

        input_size = coordinate_dim + num_sensors * num_channels
        self.inputs = keras.Input(shape=(input_size,))
        self.outputs = DNN(
            (None, input_size),
            self.num_channels,
            self.width,
            self.depth,
        )(self.inputs)

        super().__init__(inputs=self.inputs, outputs=self.outputs)


    def compile(self, optimizer):
        """Compile the model.
        
        Args:
            optimizer: optimizer
        """
        super().compile(loss="mse")
        self.loss_fn = keras.losses.MeanSquaredError()
        self.optimizer = optimizer
        self.optimizer.built = True


    def train_step(self, data):
        """Train step.

        Args:
            data: batch of observations and evaluations

        Returns:
            dict mapping metric names to current value
        """
        x, y = data

        self.optimizer.zero_grad()

        y_pred = self(x, training=True)

        loss = self.loss_fn(y, y_pred)
        loss.backward()

        self.optimizer.step()

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}