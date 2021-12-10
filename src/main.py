"""Placeholder for main
Main entry point of the program
"""
from bots.random_bot import RandomBot
from simulators.rlbot_simulator import rl_gym_simulator


def main():
    env = rl_gym_simulator.make_env()

    bot = RandomBot(env.action_space)

    for i in range(100):
        obs = env.reset()

        done = False
        while not done:
            # Here we sample a random action. If you have an agent, you would get an action from it here.
            action = bot.act(obs)

            next_obs, reward, done, gameinfo = env.step(action)

            obs = next_obs


if __name__ == '__main__':
    main()
