"""Abstract implementation of Memory"""
from abc import ABC, abstractmethod

import numpy as np


class AbstractMemory(ABC):
    """Memory stores observation and action combinations for training."""

    @abstractmethod
    def add(self, obs: np.ndarray, action: np.ndarray):
        """Add observation to memory"""

    @abstractmethod
    def get_data(self) -> (np.ndarray, np.ndarray):
        """Retrieve all samples from memory for training."""

    @abstractmethod
    def save(self) -> str:
        """Save this memory such that it can be loaded again.

        Returns
            The path needed to load this memory again.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Memory":
        """Load in a new instance from path

        Args:
            path: Path to the saved memory
                  Is also returned by save()

        Returns:
            Memory loaded from `path`
        """
