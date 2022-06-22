from __future__ import print_function, with_statement, absolute_import, division, annotations

from time import sleep

import tensorflow as tf
from tf_agents.environments import tf_py_environment

from game import GameEnv

POLICY_DIR = 'agent/policy'

if __name__ == '__main__':
    policy = tf.saved_model.load(POLICY_DIR)
    eval_env = tf_py_environment.TFPyEnvironment(GameEnv())
    time_step = eval_env.reset()
    eval_env.render('human')

    while not time_step.is_last():
        sleep(1)
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        eval_env.render('human')
