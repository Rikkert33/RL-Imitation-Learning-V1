"""Implements memory using H5File as backend.
"""
import os.path

import h5py
import numpy as np

from memory.abstract_memory import AbstractMemory


class H5FileMemory(AbstractMemory):
    """
    Class for saving rlgym observations for experiments in a H5file format
    :param experiment_name: the name of the experiment
    """

    def __init__(self, experiment_name: str):
        self.dir_experiment = os.path.join("..", "data", "obs")
        os.makedirs(self.dir_experiment, exist_ok=True)
        path_h5file = os.path.join(self.dir_experiment, f"{experiment_name}.hdf5")
        self.h5file = h5py.File(path_h5file, "a")

    def add(self, obs: np.ndarray, action: np.ndarray):
        """Add new observation, action pair to memory."""
        if (
            "observations" not in self.h5file.keys()
            and "action" not in self.h5file.keys()
        ):
            # Create the dataset at first
            self.h5file.create_dataset(
                "observations",
                data=obs.reshape(1, -1),
                chunks=True,
                maxshape=(None, *obs.shape),
            )
            self.h5file.create_dataset(
                "action",
                data=action.reshape(1, -1),
                chunks=True,
                maxshape=(None, *action.shape),
            )
        else:
            # Append new data to it
            self.h5file["observations"].resize(
                (self.h5file["observations"].shape[0] + 1), axis=0
            )
            self.h5file["observations"][-1] = obs

            self.h5file["action"].resize((self.h5file["action"].shape[0] + 1), axis=0)
            self.h5file["action"][-1] = action

    def get_data(self) -> (np.ndarray, np.ndarray):
        """
        Get all data in file
        :return:
        """
        return np.array(self.h5file["observations"]), np.array(self.h5file["action"])

    def save(self) -> str:
        pass

    @classmethod
    def load(cls, path: str) -> "H5FileMemory":
        pass
