"""Behavioral cloning bot"""

from bots.abstract_bot import AbstractBot
from bots.model_bots.abstract_model_bot import AbstractModelBot
from memory.abstract_memory import AbstractMemory


class BehavioralCloningBot(AbstractBot):
    """Base implementation for behavioral cloning bot"""

    def __init__(
        self, expert: AbstractBot, student: AbstractModelBot, memory: AbstractMemory
    ):
        self.memory = memory
        self.expert = expert
        self.student = student
        self.train_mode = True

    def act(self, obs):
        """Determines if student or expert should act"""
        if self.train_mode:
            action = self.expert.act(obs)
            self.memory.add(obs, action)
            return action

        action = self.student.act(obs)
        return action

    def train(self):
        """Train callback"""
        self.student.train(self.memory)
