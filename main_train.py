from __future__ import print_function, with_statement, absolute_import, division, annotations

import tensorflow as tf

from tf_agents.policies import policy_saver
from tf_agents.environments import tf_py_environment


from agent import *
from game import *


POLICY_DIR = 'agent/policy'
MAX_DEPTH = 1000

if __name__ == '__main__':
    print(f'{tf.version.VERSION = }')
    print(f'Devices => {tf.config.list_physical_devices()}')

    train_py_env = GameEnv(max_depth=MAX_DEPTH)
    eval_py_env = GameEnv(max_depth=MAX_DEPTH)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    agent = build_agent(train_env)

    # build replay buffer
    rb, obs = replay_buffer_observer(agent)

    # callbacks to persist trained agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_checkpointer = checkpoint_saver(agent, rb, global_step)

    train_agent(
        agent=agent,
        train_py_env=train_py_env,
        eval_env=eval_env,
        rb_observer=obs,
        replay_buffer=rb
    )

    # save until better days
    train_checkpointer.save(global_step)
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(POLICY_DIR)

