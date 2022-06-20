from __future__ import print_function, with_statement, absolute_import, division

import tensorflow as tf

from tf_agents.policies import policy_saver
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym

from agent import *
import gym_2048  # noqa

if __name__ == '__main__':
    print(f'{tf.version.VERSION = }')

    env = suite_gym.load(ENV_NAME)
    try:
        env.reset()
    except AssertionError:
        pass
    train_py_env = suite_gym.load(ENV_NAME)
    try:
        train_py_env.reset()
    except AssertionError:
        pass
    eval_py_env = suite_gym.load(ENV_NAME)
    try:
        eval_py_env.reset()
    except AssertionError:
        pass

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    agent = build_agent(env, train_env)

    # build replay buffer
    rb, obs = replay_buffer_observer(agent)

    # callbacks to persist trained agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_checkpointer = checkpoint_saver(agent, rb, global_step)

    train(
        agent=agent,
        train_py_env=train_py_env,
        eval_env=eval_env,
        rb_observer=obs,
        replay_buffer=rb
    )

    # save until better days
    train_checkpointer.save(global_step)
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save('policy')
