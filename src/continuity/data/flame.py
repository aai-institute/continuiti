import numpy as np
import pandas as pd
from continuity.data.dataset import DataSet, Sensor, Observation


class Flame(DataSet):
    """Turbulent flow samples from flame dataset"""

    def __init__(self, size, batch_size):
        self.num_sensors = 16 * 16
        self.size = size

        self.coordinate_dim = 2
        self.num_channels = 1

        # Generate observations
        observations = [
            self._generate_observation(i)
            for i in range(self.size)
        ]
        
        super().__init__(observations, batch_size)

    def _generate_observation(self, i: int):
        # Load data
        input_path = 'flame/'
        res = 'LR'

        df = pd.read_csv(input_path + 'train.csv')
        data_path = input_path + f"flowfields/{res}/train"

        filename = df[f'ux_filename'][i+1]
        u = np.fromfile(
            data_path + "/" + filename, dtype="<f4"
        ).reshape(16, 16, 1)

        # Normalize
        u = (u - u.mean()) / u.std()

        # Positions
        x = np.stack(
            np.meshgrid(
                np.linspace(-1, 1, 16),
                np.linspace(-1, 1, 16),
            ),
            axis=2
        )

        # Sensors
        sensors = [
            Sensor(x[i][j], u[i][j])
            for i in range(16) for j in range(16)
        ]
        return Observation(sensors)

