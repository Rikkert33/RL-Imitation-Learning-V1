"""Model Bot used for the straight line scenario
"""
import os.path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from bots.model_bots.abstract_model_bot import AbstractModelBot
from memory.abstract_memory import AbstractMemory


class StraightLineModelBot(AbstractModelBot):
    """Simple bot that can only control the throttle"""

    def __init__(self):
        self.model = self.create_model()
        self.model_dir = os.path.join("..", "output", self.__class__.__name__)
        self.model_path = os.path.join(self.model_dir, "model")
        os.makedirs(self.model_path, exist_ok=True)

    @staticmethod
    def create_model():
        """Initializes the model"""
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(70,)),  # obs is 70
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(
                    1, activation="tanh"
                ),  # output is the amount of gas the bot gives between -1 and 1
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mse"])

        return model

    def train(self, memory: AbstractMemory):
        """Train callback
        Trains the internal model.
        """
        observations, actions = memory.get_data()

        expert_obs, expert_actions = observations, actions[:, 0]
        assert (
            expert_obs.shape[0] == expert_actions.shape[0]
        ), "unequal input size expert_obs and expert_actions"

        (
            expert_obs_train,
            expert_obs_test,
            expert_actions_train,
            expert_actions_test,
        ) = train_test_split(expert_obs, expert_actions, test_size=0.2)

        self.model.fit(expert_obs_train, expert_actions_train, epochs=20)

        self.model.evaluate(expert_obs_test, expert_actions_test)

    def evaluate(self, memory: AbstractMemory):
        """Evaluation callback

        Evaluate based on the given memory
        """

    def save(self):
        """
        Save the keras model
        :return:
        """
        self.model.save(self.model_path)

    def load(self):
        """
        Load the keras model from directory
        :return:
        """
        self.model = tf.keras.models.load_model(self.model_path)

    def act(self, obs):
        """Act based on given obs"""
        gas = tf.squeeze(self.model(tf.expand_dims(obs, 0)))
        return np.asarray([gas, 0, 0, 0, 0, 0, 0, 0])
