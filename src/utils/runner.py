"""Helper functions to run games
"""
import gym
import tqdm

from bots.abstract_bot import AbstractBot


def play_games(
    env: gym.Env,
    bot: AbstractBot,
    n_games: int,
):
    """Play multiple games back-to-back

    Args:
        env: gym.Env to play in.
        bot: AbstractBot that provides the actions given the obs from the env.
        n_games: The number of games to play.


    Returns:

    """
    total_reward = 0
    for _ in tqdm.tqdm(range(n_games)):
        total_reward += play_game(env, bot)
    return total_reward / n_games


def play_game(env: gym.Env, bot: AbstractBot):
    """Play a single game in the env.

    Args:
        env: gym.Env to play in.
        bot: AbstractBot that provides the actions given the obs from the env.

    Returns:
        total_reward: int representing the total reward received from the env.
    """
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = bot.act(obs)
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
    return total_reward
