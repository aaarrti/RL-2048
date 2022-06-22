from __future__ import print_function, with_statement, absolute_import, division

import tensorflow as tf

from tf_agents.policies import policy_saver
from tf_agents.environments import tf_py_environment
import click
from time import sleep

from agent import *
from game import *

POLICY_DIR = 'policy'
STEP_DEPTH = 100


@click.command()
@click.option('--train', default=False)
@click.option('--play', default=False)
def main(train, play):
    if train:
        train_main()
    if play:
        play_main()


def train_main():
    print(f'{tf.version.VERSION = }')

    train_py_env = GameEnv()
    eval_py_env = GameEnv()

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


def play_main():
    policy = tf.saved_model.load(POLICY_DIR)
    eval_env = tf_py_environment.TFPyEnvironment(GameEnv())
    time_step = eval_env.reset()
    eval_env.render('human')

    while not time_step.is_last():
        #sleep(1)
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        eval_env.render('human')


if __name__ == '__main__':
    main()
