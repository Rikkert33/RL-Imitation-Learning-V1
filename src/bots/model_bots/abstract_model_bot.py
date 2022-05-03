"""
Abstract class for model based bots
"""
import abc

from bots.abstract_bot import AbstractBot
from memory.abstract_memory import AbstractMemory


class AbstractModelBot(AbstractBot, abc.ABC):
    """
    Abstract class for model based bots
    """

    @abc.abstractmethod
    def train(self, memory: AbstractMemory):
        """Training of the model"""

    @abc.abstractmethod
    def save(self):
        """
        Save the model to standard directory
        :return:
        """

    @abc.abstractmethod
    def load(self):
        """
        Load the model from default directory
        :return:
        """
