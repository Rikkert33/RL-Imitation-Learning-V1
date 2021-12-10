"""Abstract Interface for a bot implementation
PLACEHOLDER
Each bot should use this interface so it can interact with a simulator.
Currently the bot should be fully contained in a subfolder in this directory.
In the future we might want to separate the 'learning' part from the
'playing/interacting' part of a bot. However currently we keep it contained.
"""
from abc import ABC, abstractmethod


class AbstractBot(ABC):
    @abstractmethod
    def act(self, obs):
        """Return action based on observation
        """
