from __future__ import print_function, with_statement, absolute_import, division

from tf_agents.policies import policy_saver
from tf_agents.environments import tf_py_environment
import tensorflow as tf

from agent import *


if __name__ == '__main__':
    #os.environ['GRPC_TRACE'] = 'all'
    #os.environ['GRPC_VERBOSITY'] = 'debug'
    # provision envs
    py_env = provision_env()
    train_py_env = provision_env()
    eval_py_env = provision_env()

    # convert to TCF env
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # build agent
    counter = tf.Variable(0)
    agent = build_agent(train_step_counter=counter, env=py_env)

    # build replay buffer
    rb, obs = replay_buffer_observer(agent)
    ds = as_dataset(rb)

    # callbacks to persist trained agent
    global_step = tf.compat.v1.train.get_or_create_global_step()
    train_checkpointer = checkpoint_saver(agent, rb, global_step)

    train(
        agent=agent,
        dataset=ds,
        train_py_env=train_env,
        eval_env=eval_env,
        env=py_env,
        rb_observer=obs
    )

    # save until better days
    train_checkpointer.save(global_step)
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save('../policy')
