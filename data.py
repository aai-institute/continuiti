import math
import numpy as np
from numpy import ndarray
import keras_core as keras
from typing import List, Optional


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


    def get_observation(self, idx: int) -> Observation:
        """Return observation at index"""
        return self.observations[idx]


    def __len__(self) -> int:
        """Return number of batches"""
        num_obs = len(self.observations)
        num_sensors = len(self.observations[0].sensors)
        return math.ceil(num_obs * num_sensors / self.batch_size)
    

    def __getitem__(self, idx: int):
        """Return batch of observations"""
        obs_pos = []
        evals = []

        for _ in range(self.batch_size):
            # Select random observation
            obs_idx = np.random.randint(len(self.observations))
            observation = self.observations[obs_idx]

            # Select random sensor
            sen_idx = np.random.randint(len(observation.sensors))

            # Take one sensor as label
            sensor = observation.sensors[sen_idx]
            x, u = sensor.x, sensor.u

            # Append observation, x and u
            obs_pos.append(self.flatten(observation, x))
            evals.append(u)

        return np.array(obs_pos), np.array(evals)
    

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

