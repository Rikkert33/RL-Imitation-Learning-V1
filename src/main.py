"""Placeholder for main
Main entry point of the program
"""

from bots.behavioral_cloning_bot import BehavioralCloningBot
from bots.model_bots.straight_line_model_bot import StraightLineModelBot
from bots.straight_line_bot import StraightLineBot
from memory import H5FileMemory

from simulators.rlbot_simulator import rl_gym_simulator
from utils.runner import play_games


def main():
    """Entrypoint of the program"""
    env = rl_gym_simulator.make_env(game_speed=100)
    try:
        expert = StraightLineBot(action_space=env.action_space)
        student = StraightLineModelBot()
        student.load()
        memory = H5FileMemory(expert.__class__.__name__)
        bot: BehavioralCloningBot = BehavioralCloningBot(expert, student, memory)

        print(f"Expert average score: {play_games(env, bot, 10)}")

        bot.train()
        bot.student.save()

        bot.train_mode = False
        print("Evaluating")
        print(f"Evaluated average score: {play_games(env, bot, 100)}")

        print("Done simulation")
    finally:
        env.close()


if __name__ == "__main__":
    main()
