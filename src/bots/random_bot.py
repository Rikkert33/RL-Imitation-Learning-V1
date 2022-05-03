"""Random RL Bot
"""

from bots.abstract_bot import AbstractBot


class RandomBot(AbstractBot):
    """Bot that returns random actions sampled from the action_space."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):  # pylint: disable=unused-argument
        return self.action_space.sample()
