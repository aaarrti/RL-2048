from tf_agents.environments import TFPyEnvironment, PyEnvironment
from tf_agents.policies import TFPolicy
from tf_agents.replay_buffers import ReverbAddTrajectoryObserver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.drivers import py_driver

from .constants import *


def compute_avg_return(environment: TFPyEnvironment, policy: TFPolicy):
    total_return = 0.0
    for _ in range(NUM_EVAL_EPISODES):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / NUM_EVAL_EPISODES
    return avg_return.numpy()[0]


def collect_episode(environment: PyEnvironment, policy, rb_observer: ReverbAddTrajectoryObserver):
    driver = py_driver.PyDriver(
        environment,
        py_tf_eager_policy.PyTFEagerPolicy(
            policy, use_tf_function=True),
        [rb_observer],
        max_episodes=COLLECT_STEPS_PER_EPOCH)
    initial_time_step = environment.reset()
    driver.run(initial_time_step)
