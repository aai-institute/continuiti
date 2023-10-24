import math
import numpy as np
from numpy import ndarray
import keras_core as keras
from typing import List, Optional
import pandas as pd

class Sensor:
    """Sensor
    
    A sensor is tuple (x, u).
    """
    def __init__(self, x: ndarray, u: ndarray):
        self.x = x
        self.u = u

        self.coordinate_dim = x.shape[0]
        self.num_channels = u.shape[0]


class Observation:
    """Observation
    
    An observation is a list of N sensors.
    """
    def __init__(self, sensors: List[Optional[Sensor]]):
        self.sensors = sensors


class DataSet(keras.utils.PyDataset):
    """DataSet
    
    A data set is a set of observations.
    """
    def __init__(
            self,
            observations: List[Observation],
            batch_size: int,
            **kwargs
        ):
        super().__init__(**kwargs)

        self.observations = observations
        self.batch_size = batch_size

        # Build data tensors x and y
        obs_pos = []
        evals = []

        for i in range(len(self.observations)):
            observation = self.observations[i]

            for j in range(len(observation.sensors)):
                # Take one sensor as label
                sensor = observation.sensors[j]
                x, u = sensor.x, sensor.u
                ox = self.flatten(observation, x)
                obs_pos.append(ox)
                evals.append(u)

        self.x = np.array(obs_pos)
        self.y = np.array(evals)


    def get_observation(self, idx: int) -> Observation:
        """Return observation at index"""
        return self.observations[idx]


    def __len__(self) -> int:
        """Return number of batches"""
        return math.ceil(len(self.x) / self.batch_size)
    

    def __getitem__(self, idx: int):
        """Return batch of observations"""
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        return self.x[low:high], self.y[low:high]
    

    def flatten(self, observation: Observation, position: ndarray) -> ndarray:
        """Convert observation and position into ndarray.
        
        Args:
            observation: observation object with s sensors with u of shape (c,)
            position: a single position p of shape (d,)

        Returns:
            tensor: a tensor of shape (s * c + d,)
                that contains (u_1, u_2, ... u_s, p)
        """
        s = self.num_sensors
        d = self.coordinate_dim
        c = self.num_channels
        assert position.shape == (d,)

        # Concatenate u for each sensor
        tensor = np.zeros((s * c + d,))
        for i, sensor in enumerate(observation.sensors):
            u = sensor.u
            assert u.shape == (c,)
            tensor[i * c : (i+1) * c] = u

        # Append position
        tensor[-d:] = position

        return tensor


class SineWaves(DataSet):
    """Sine waves"""

    def __init__(self, num_sensors, size, batch_size):
        self.num_sensors = num_sensors
        self.size = size

        self.coordinate_dim = 1
        self.num_channels = 1

        # Generate observations
        observations = [
            self._generate_observation(i)
            for i in range(self.size)
        ]
        
        super().__init__(observations, batch_size)

    def _generate_observation(self, i: int):
        x = np.linspace(-1, 1, self.num_sensors)
        u = np.sin((1+i) * np.pi * x)
        sensors = [
            Sensor(np.array([x]), np.array([u]))
            for x, u in zip(x, u)
        ]
        return Observation(sensors)



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

