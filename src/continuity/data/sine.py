import numpy as np
from continuity.data.dataset import DataSet, Sensor, Observation


class SineWaves(DataSet):
    """Sine waves"""

    def __init__(self, num_sensors, size, batch_size):
        self.num_sensors = num_sensors
        self.size = size

        self.coordinate_dim = 1
        self.num_channels = 1

        # Generate observations
        observations = [
            self._generate_observation(i / max(1, self.size - 1))
            for i in range(self.size)
        ]

        super().__init__(observations, batch_size)

    def _generate_observation(self, i: int):
        x = np.linspace(-1, 1, self.num_sensors)
        u = np.sin((1 + i) * np.pi * x)
        sensors = [Sensor(np.array([x]), np.array([u])) for x, u in zip(x, u)]
        return Observation(sensors)
