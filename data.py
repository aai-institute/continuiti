import math
from copy import deepcopy
import numpy as np
from numpy import ndarray
import keras_core as keras

class Observation:
    """Observation
    
    An observation is a list of N sensor tuples (x, u) 
    """
    def __init__(self, sensors: list):
        self.sensors = sensors

        x, u = sensors[0]
        self.coordinate_dim = x.shape[0]
        self.num_channels = u.shape[0]
        self.num_sensors = len(sensors)


class DataSet(keras.utils.PyDataset):
    """DataSet
    
    A data set is a set of observations.
    """
    def __init__(self, observations: list, batch_size: int, **kwargs):
        """A data set is a set of observations."""
        super().__init__(**kwargs)

        self.observations = observations
        self.batch_size = batch_size

        # Extract dimension of coordinate space, channels and sensors
        obs = observations[0]
        self.coordinate_dim = obs.coordinate_dim
        self.num_channels = obs.num_channels
        self.num_sensors = obs.num_sensors

    def get_observation(self, idx: int) -> Observation:
        """Return observation at index"""
        return self.observations[idx]

    def flatten(self, observation: Observation, position: ndarray) -> ndarray:
        """Convert observation and position into ndarray.
        
        Args:
            observation: a single observation object with s sensors, 
                each sensor is a tuple of (x, u) 
                where x is of shape (d,) and u of shape (c,)
            position: a single position of shape (d,)

        Returns:
            tensor: a tensor of shape (s * (d + c) + d,)
        """
        s = self.num_sensors
        d = self.coordinate_dim
        c = self.num_channels
        assert position.shape == (d,)

        # Concatenate x and u for each sensor
        tensor = np.zeros((s * (d + c) + d,))
        for i, sensor in enumerate(observation.sensors):
            x, u = sensor
            assert x.shape == (d,)
            assert u.shape == (c,)
            tensor[i : i + (d + c)] = np.concatenate((x, u))

        # Append position
        tensor[-d:] = position

        return tensor


    def __len__(self) -> int:
        """Return number of batches"""
        return len(self.observations)
    

    def __getitem__(self, idx: int):
        """Return batch of observations"""
        observation = self.observations[idx]

        # Convert batch of observations to batch of tensors
        obs_pos = []
        evals = []

        for mask_idx in range(self.num_sensors):
            obs = deepcopy(observation)

            # Extract x and u to generate label
            x, u = obs.sensors[mask_idx]

            # Mask sensor mask_idx by duplicating another sensor
            rand_idx = np.random.randint(self.num_sensors)
            obs.sensors[mask_idx] = obs.sensors[rand_idx]

            # Append observation/position and evaluation
            obs_pos.append(self.flatten(obs, x))
            evals.append(u)
        
        return np.array(obs_pos), np.array(evals)
    

class SineWaves(DataSet):
    """Sine waves"""

    def __init__(self, num_sensors, size):
        self.num_sensors = num_sensors
        self.size = size

        # Generate observations
        observations = [
            self._generate_observation(i)
            for i in range(self.size)
        ]
        
        super().__init__(observations, num_sensors)

    def _generate_observation(self, i: int):
        x = np.linspace(0, 1, self.num_sensors)
        u = np.sin((2+i) * np.pi * x)
        sensors = [
            (np.array([x]), np.array([u]))
            for x, u in zip(x, u)
        ]
        return Observation(sensors)

