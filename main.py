from __future__ import print_function, with_statement, absolute_import, division

from tf_agents.policies import policy_saver
from tf_agents.environments import tf_py_environment
import tensorflow as tf

from agent import *
from tf_agents.environments import suite_gym
import gym_2048 # noqa
import gym
from tf_agents.specs import tensor_spec

ENV_NAME = 'CartPole-v0'


if __name__ == '__main__':

    # convert to TCF env
    env = suite_gym.load(ENV_NAME)
    train_py_env = suite_gym.load(ENV_NAME)
    eval_py_env = suite_gym.load(ENV_NAME)

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # build agent
    counter = tf.Variable(0)

    q_net = build_q_net(env)
    agent = build_agent(train_step_counter=counter, train_env=train_env, q_net=q_net)

    # build replay buffer
    rb, obs = replay_buffer_observer(agent)
    ds = as_dataset(rb)

    # callbacks to persist trained agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_checkpointer = checkpoint_saver(agent, rb, global_step)

    train(
        agent=agent,
        dataset=ds,
        train_py_env=train_py_env,
        eval_env=eval_env,
        env=env,
        rb_observer=obs
    )

    # save until better days
    train_checkpointer.save(global_step)
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save('policy')
