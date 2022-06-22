from typing import Any, Text, Optional

import tf_agents.specs
from tf_agents.environments import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from .game import Game
from .util import *
import numpy as np
from threading import Thread, Lock


class GameEnv(PyEnvironment):

    moves_depth = 0
    MAX_MOVES_DEPTH = 10
    mutex = Lock()

    def __init__(self):
        super().__init__()
        self.game = Game()

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
        old_score = self.game.score
        move = map_int_to_key(action)
        self.game.do_move(move)
        new_score = self.game.score
        observation = np.asarray(self.game.observation).flatten()
        if self.stuck or self.moves_depth == self.MAX_MOVES_DEPTH:
            return ts.termination(observation=observation, reward=old_score - new_score)
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
