"""Simple memory class storing data in lists
"""
import numpy as np

from memory.abstract_memory import AbstractMemory


class SimpleMemory(AbstractMemory):
    """Simple memory that does not implement loading and saving"""

    def __init__(self):
        self._observations = []
        self._actions = []

    def add(self, obs: np.ndarray, action: np.ndarray):
        self._observations.append(obs)
        self._actions.append(action)

    def get_data(self) -> (np.ndarray, np.ndarray):
        return np.array(self._observations), np.array(self._actions)

    def save(self) -> str:
        raise NotImplementedError("SimpleMemory is not savable")

    @classmethod
    def load(cls, path: str) -> "SimpleMemory":
        raise NotImplementedError("SimpleMemory is not loadable")
