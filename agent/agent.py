import tensorflow as tf
from tf_agents.agents import TFAgent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.utils import common
from tqdm import tqdm
from tf_agents.replay_buffers import ReverbAddTrajectoryObserver, ReverbReplayBuffer
from tf_agents.drivers import py_driver
from tf_agents.environments import TFPyEnvironment, PyEnvironment

from .config import *
from .metrics import *
from tf_agents.specs import tensor_spec


def dense_layer(num_units: int):
    kernel = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
    return tf.keras.layers.Dense(num_units,
                                 activation=tf.keras.activations.relu,
                                 kernel_initializer=kernel
                                 )


def build_q_net(env: PyEnvironment, fc_layer_params=(100, 50)):
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    return q_net


def build_agent(train_step_counter: tf.Variable, train_env: TFPyEnvironment, q_net):
    ag = dqn_agent.DqnAgent(train_env.time_step_spec(),
                            train_env.action_spec(),
                            q_network=q_net,
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            td_errors_loss_fn=common.element_wise_squared_loss,
                            train_step_counter=train_step_counter
                            )
    ag.initialize()
    return ag


def train(agent: TFAgent,
          dataset: tf.data.Dataset,
          train_py_env: PyEnvironment,
          eval_env: TFPyEnvironment,
          env: PyEnvironment,
          rb_observer: ReverbAddTrajectoryObserver
          ):
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    # Reset the environment.
    time_step = train_py_env.reset()

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        env,
        py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True),
        [rb_observer],
        max_steps=collect_steps_per_iteration)

    iterator = iter(dataset)

    for _ in tqdm(range(num_iterations)):

        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)


def checkpoint_saver(agent: TFAgent,
                     replay_buffer: ReverbReplayBuffer,
                     global_step
                     ):
    train_checkpointer = common.Checkpointer(
        ckpt_dir='checkpoints',
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    return train_checkpointer
