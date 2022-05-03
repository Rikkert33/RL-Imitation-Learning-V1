"""Logic for the straight line bot

Drives the bot in a straight line to a `target_location`

Currently, only works in 1 Dimension
"""
import numpy as np

from bots.abstract_bot import AbstractBot


class StraightLineBot(AbstractBot):
    """Drives a bot in a straight line towards the target location."""

    def __init__(self, action_space, target_location=(0, 0, 0)):
        self.action_space = action_space
        self.target_location = target_location
        self.position_idx = [51, 52, 53]
        self.speed_idx = [60, 61, 62]

    def act(self, obs) -> np.ndarray:
        """Returns the action to move to the target location

        Args:
            obs:
             ndarray representing the current observation

        Returns:
            ndarray representing the action to be done
        """
        # Take y components of location and speed data
        position = obs[self.position_idx][1]
        speed = obs[self.speed_idx][1]

        distance_to_target = self.target_location[1] - position

        error = 4 * distance_to_target
        speed = np.sign(distance_to_target + speed) * speed

        gas = np.clip(error + speed, -1, 1)

        return np.asarray([gas, 0, 0, 0, 0, 0, 0, 0])
