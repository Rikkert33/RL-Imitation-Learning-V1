"""
Save the observations to file
"""
import os.path
from datetime import datetime

import numpy as np

from memory.abstract_memory import AbstractMemory


class FileBasedMemory(AbstractMemory):
    """
    Class for saving rlgym observations for experiments
    Automatically makes the required directory and increments run number
    :param experiment_name: the name of the experiment
    """

    def __init__(self, experiment_name: str):
        self.dir_experiment = os.path.join("..", "data", "obs", experiment_name)
        os.makedirs(self.dir_experiment, exist_ok=True)
        self.observations = []
        self.actions = []

    def add(self, obs: np.ndarray, action: np.ndarray):
        """
        Add an observation to save later
        :param action: optional numpy array of action
        :param obs: numpy array of observation
        """
        self.observations.append(obs)
        self.actions.append(action)

    def save(self, clear_obs: bool = True):
        """
        Save the recorded observations to a file
        :type clear_obs: bool if to clear observations after saving
        """
        path_file = os.path.join(
            self.dir_experiment, f'run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        )

        obs = np.stack(self.observations)
        actions = np.stack(self.actions)
        np.savez_compressed(path_file, obs=obs, actions=actions)

        if clear_obs:
            self._clear_observations()

    @classmethod
    def load(cls, path: str) -> "FileBasedMemory":
        pass

    def _clear_observations(self):
        """
        Clear current hold observations
        :return:
        """
        self.observations = []
        self.actions = []
