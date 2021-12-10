from bots.abstract_bot import AbstractBot


class RandomBot(AbstractBot):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs):
        return self.action_space.sample()
