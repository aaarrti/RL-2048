from typing import Any
import tensorflow as tf
import tf_agents as tfa
from tf_agents.environments import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.typing import types
from google.protobuf.empty_pb2 import Empty
import grpc

from .config import game_base_uri, env_provisioner_uri
from proto.env_pb2_grpc import GameEnvServiceStub
from proto.game_pb2_grpc import GameServiceStub
from proto.game_pb2 import MoveMessage
from proto.env_pb2 import GameMessage
from .util import *


class Game2048Env(PyEnvironment):
    pid: int
    port: int

    def __init__(self, pid, port):
        super().__init__()
        self.pid = pid
        self.port = port

    def observation_spec(self) -> types.NestedArraySpec:
        return tfa.specs.ArraySpec(shape=(4, 4), dtype=tf.int64, name='observation')

    def reward_spec(self) -> types.NestedArraySpec:
        return tfa.specs.ArraySpec(shape=(), dtype=tf.float64, name='reward')

    def get_info(self) -> types.NestedArray:
        pass

    @log_after
    def get_state(self) -> Any:
        pass

    @log_before
    def set_state(self, state: Any) -> None:
        pass

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        """
        Updates the environment according to action and returns a `TimeStep`.
        """
        with grpc.insecure_channel(game_base_uri + self.port) as channel:
            stub = GameServiceStub(channel)
            res = stub.doMove(MoveMessage(value=action))
            return TimeStep(reward=res.reward, observation=res.observation)

    def _reset(self) -> ts.TimeStep:
        with grpc.insecure_channel(game_base_uri + self.port) as channel:
            stub = GameServiceStub(channel)
            res = stub.resetGame(Empty)
            return TimeStep(reward=res.reward, observation=res.observation)

    def action_spec(self) -> types.NestedArraySpec:
        """
        The `action_spec()` method returns the shape, data types, and allowed values of valid actions.
        """
        return tfa.specs.BoundedArraySpec(shape=(), dtype=tf.int64, name='action', minimum=0, maximum=3)

    def time_step_spec(self) -> TimeStep:
        """
        The `time_step_spec()` method returns the specification for the `TimeStep` tuple.
        Its `observation` attribute shows the shape of observations,
        the data types, and the ranges of allowed values.
        The `reward` attribute shows the same details for the reward.
        """
        return TimeStep(reward=self.reward_spec(), observation=self.observation_spec())


def provision_env() -> Game2048Env:
    with grpc.insecure_channel(env_provisioner_uri, options=(('grpc.enable_http_proxy', 0),)) as channel:
        stub = GameEnvServiceStub(channel)
        res = stub.provisionEnvironment(Empty())
        game = Game2048Env(pid=res.pid, port=res.port)
        return game


def stop_env(game: Game2048Env):
    with grpc.insecure_channel(env_provisioner_uri) as channel:
        stub = GameEnvServiceStub(channel)
        stub.stopEnvironment(GameMessage(port=game.pid))
