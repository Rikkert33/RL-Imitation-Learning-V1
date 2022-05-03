"""Scenario for the straight line task

Straight line task:
Drive the car in a straight line to the center of the field.
"""
import math
import random

import numpy as np
from rlgym.utils import StateSetter
from rlgym.utils.common_values import BACK_WALL_Y
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.state_setters import StateWrapper


# pylint: disable=unused-argument, no-self-use
class StraightLineStateSetter(StateSetter):
    """Setup the rocket league environment for a straight line experiment.

    Sets the following properties:
     * Random y location inline with the center (goal to goal) line.
     * Rotates the car towards one goal (exactly on the goal to goal line).
     * Moves the bal outside the field.
    """

    def reset(self, state_wrapper: StateWrapper):  # pylint: disable=no-self-use
        """Reset the environment according to this state."""
        y = random.randint(-BACK_WALL_Y, BACK_WALL_Y)

        state_wrapper.cars[0].set_pos(0, y, 0)
        state_wrapper.cars[0].set_rot(0, math.pi / 2, 0)

        state_wrapper.ball.set_pos(-2000, 0, 0)


class StraightLineReward(RewardFunction):
    """Simple reward function for the straight line scenario."""

    def reset(self, initial_state: GameState):
        """Reset callback, unused."""

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        """Calculate reward based on the distance to the center.
        Args:
            player:
            state:
            previous_action:

        Returns:
            reward
        """
        position = player.car_data.position
        y = position[1]
        reward = -abs(y)
        return reward
