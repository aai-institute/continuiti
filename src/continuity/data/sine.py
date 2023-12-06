import numpy as np
from continuity.data.dataset import SelfSupervisedDataSet, Sensor, Observation


class SineWaves(SelfSupervisedDataSet):
    r"""Creates a data set of sine waves.

    The data set is generated by sampling sine waves at the given number of
    sensors placed evenly in the interval $[-1, 1]$. The wave length of the sine waves is evenly distributed between $\pi$ for the first observation and $2\pi$ for the last observation, respectively.

    Args:
        num_sensors: Number of sensors.
        size: Size of data set.
        batch_size: Batch size. Defaults to 32.
    """

    def __init__(self, num_sensors: int, size: int, batch_size: int = 32):
        self.num_sensors = num_sensors
        self.size = size

        self.coordinate_dim = 1
        self.num_channels = 1

        # Generate observations
        observations = [self.generate_observation(i) for i in range(self.size)]

        super().__init__(observations, batch_size)

    def generate_observation(self, i: float):
        """Generate observation

        Args:
            i: Index of observation (0 <= i <= size).
        """
        x = np.linspace(-1, 1, self.num_sensors)

        if self.size == 1:
            w = 1
        else:
            w = i / self.size + 1

        u = np.sin(w * np.pi * x)

        sensors = [Sensor(np.array([x]), np.array([u])) for x, u in zip(x, u)]

        return Observation(sensors)
