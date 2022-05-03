"""Simulator implementation for the RLGym api.

Requires Windows and bakkesmod to run.
"""

import rlgym

from simulators.scenarios.straigth_line_scenario import (
    StraightLineStateSetter,
    StraightLineReward,
)


def make_env(**kwargs):
    """Create a gym environment for rocket league

    Args:
        **kwargs:

    Returns:
        Gym env for Rocket League
    """
    return rlgym.make(
        **kwargs, state_setter=StraightLineStateSetter(), reward_fn=StraightLineReward()
    )
