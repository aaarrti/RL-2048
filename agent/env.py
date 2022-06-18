from typing import Any

import numpy as np
import tensorflow as tf
import tf_agents as tfa
from tf_agents.environments import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


from .util import *


def max_arr(arr):
    return np.max(np.array(arr))


class Game2048Env(PyEnvironment):
    port: int

    def __init__(self, port):
        super().__init__()
        self.port = port

    def _uri(self):
        return game_base_uri + str(self.port)

    @log_before
    def observation_spec(self) -> types.NestedArraySpec:
        return tfa.specs.BoundedTensorSpec(shape=(16,), dtype=tf.int64, name='observation', minimum=2, maximum=2048)

    @log_before
    def reward_spec(self) -> types.NestedArraySpec:
        return tfa.specs.TensorSpec(shape=(), dtype=tf.int64, name='reward')

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
    @log_after
    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        """
        Updates the environment according to action and returns a `TimeStep`.
        """
        pass

    @log_before
    @log_after
    def _reset(self) -> ts.TimeStep:
        pass

    @log_before
    def action_spec(self) -> types.NestedArraySpec:
        """
        The `action_spec()` method returns the shape, data types, and allowed values of valid actions.
        FIXME not all actions are always available
        """
        return tfa.specs.BoundedArraySpec(shape=(), dtype=int, name='action', minimum=0, maximum=3)
