from __future__ import print_function, with_statement, absolute_import, annotations


from typing import Any, Text, Optional

import numpy as np
import tf_agents.specs
from tf_agents.environments import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from .game import Game
from util import *


class GameEnv(PyEnvironment):

    moves_depth = 0

    def __init__(self, max_depth=None):
        """
        :param max_depth: limit depth of moves for training
        """
        super().__init__()
        self.game = Game()
        self.max_depth = max_depth

    @log_after
    def observation_spec(self) -> types.NestedArraySpec:
        """
        In 2048 observation is a 2d array, namely board itself
        :return: 
        """
        return tf_agents.specs.BoundedArraySpec(shape=(16,), minimum=0, dtype=int, name='observation_spec')

    @log_after
    def action_spec(self) -> types.NestedArraySpec:
        return tf_agents.specs.BoundedArraySpec(shape=(), minimum=0, maximum=3, dtype=int, name='action_spec')

    @log_before
    def get_info(self) -> types.NestedArray:
        pass

    @log_after
    def get_state(self) -> Any:
        pass

    @log_before
    def set_state(self, state: Any) -> None:
        pass

    #@log_before
    #@log_after
    def _step(self, action: types.NestedArray) -> ts.TimeStep:

        move = map_int_to_key(action)

        old_score = self.game.score

        old_state = flatten(self.game.game.grid)
        self.game.do_move(move)

        new_state = flatten(self.game.game.grid)
        new_score = self.game.score

        observation = np.asarray(self.game.observation).flatten()

        if old_state == new_state:
            # punish agent fot not moving cells
            return ts.truncation(observation=observation, reward=-1)

        max_depth_reached = self.max_depth is not None and self.moves_depth == self.max_depth
        if self.stuck or max_depth_reached:
            return ts.truncation(observation=observation, reward=old_score - new_score)
        else:
            self.moves_depth = self.moves_depth + 1
            return ts.transition(observation=observation, reward=old_score - new_score)

    #@log_before
    #@log_after
    def _reset(self) -> ts.TimeStep:
        self.game.reset()
        observation = np.asarray(self.game.observation).flatten()
        return ts.restart(observation=observation)

    @property
    def stuck(self):
        return self.game.stuck

    def render(self, mode: Text = 'rgb_array') -> Optional[types.NestedArray]:
        return self.game.render()


#@log_before
#@log_after
def map_int_to_key(n: int) -> str:
    if n == 0:
        return 'w'
    if n == 1:
        return 'a'
    if n == 2:
        return 's'
    if n == 3:
        return 'd'


#@log_before
#@log_after
def map_key_to_int(k: str) -> int:
    if k == 'w':
        return 0
    if k == 'a':
        return 1
    if k == 's':
        return 2
    if k == 'd':
        return 3
