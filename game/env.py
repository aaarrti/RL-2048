from typing import Any

import tf_agents.specs
from tf_agents.environments import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
import tensorflow as tf
from tf_agents.environments import suite_gym

from game import Game
from util import *


class GameEnv(PyEnvironment):

    def __init__(self):
        super().__init__()
        self.game = Game()

    def observation_spec(self) -> types.NestedArraySpec:
        """
        In 2048 observation is a 2d array, namely board itself
        :return: 
        """
        return tf_agents.specs.BoundedTensorSpec(shape=(4, 4), minimum=0, dtype=tf.int32, maximum=2048)

    @log_after
    def action_spec(self) -> types.NestedArraySpec:
        moves = self.game.get_all_possible_moves()
        int_moves = [map_key_to_int(i) for i in moves]
        return tf.TensorSpec.from_tensor(tf.constant(int_moves))

    @log_before
    def get_info(self) -> types.NestedArray:
        pass

    @log_after
    def get_state(self) -> Any:
        pass

    @log_before
    def set_state(self, state: Any) -> None:
        pass

    @log_before
    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        old_score = self.game.score
        move = map_int_to_key(action.numpy())
        self.game.do_move(move)
        new_score = self.game.score
        return ts.transition(observation=self.game.observation, reward=old_score - new_score)

    @log_before
    def _reset(self) -> ts.TimeStep:
        self.game.reset()
        return ts.restart(observation=self.game.observation)


@log_before
@log_after
def map_int_to_key(n: int) -> str:
    if n == 0:
        return 'w'
    if n == 1:
        return 'a'
    if n == 2:
        return 's'
    if n == 3:
        return 'd'


@log_before
@log_after
def map_key_to_int(k: str) -> int:
    if k == 'w':
        return 0
    if k == 'a':
        return 1
    if k == 's':
        return 2
    if k == 'd':
        return 3


if __name__ == '__main__':
    gym = suite_gym.load('CartPole-v0')

    env = GameEnv()

    env.observation_spec()
    env.action_spec()

    env._reset()

    print()
