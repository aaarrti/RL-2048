import tensorflow as tf
from tf_agents.agents import TFAgent
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential

from .metrics import *
from tf_agents.utils import common
from tqdm import tqdm
from tf_agents.replay_buffers import ReverbAddTrajectoryObserver, ReverbReplayBuffer

from tf_agents.environments import PyEnvironment, TFPyEnvironment

from .config import *
from .metrics import *
from tf_agents.specs import tensor_spec



def dense_layer(num_units: int):
    kernel = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
    return tf.keras.layers.Dense(num_units,
                                 activation=tf.keras.activations.relu,
                                 kernel_initializer=kernel
                                 )


def build_q_net(env: PyEnvironment):
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    dense_layers = [dense_layer(num_units) for num_units in FC_LAYERS_PARAMETERS]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    q_net = sequential.Sequential(dense_layers + [q_values_layer])
    return q_net


def build_agent(train_env: TFPyEnvironment, q_net: tf.keras.Model):
    ag = dqn_agent.DqnAgent(train_env.time_step_spec(),
                            train_env.action_spec(),
                            q_network=q_net,
                            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                            td_errors_loss_fn=common.element_wise_squared_loss,
                            train_step_counter=tf.Variable(0)
                            )
    ag.initialize()
    return ag





def train(agent: TFAgent,
          train_py_env: PyEnvironment,
          eval_env: TFPyEnvironment,
          rb_observer: ReverbAddTrajectoryObserver,
          replay_buffer: ReverbReplayBuffer
          ):
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train, jit_compile=True)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy)
    returns = [avg_return]

    for _ in tqdm(range(EPOCHS)):
        # Collect a few episodes using collect_policy and save to the replay buffer.
        collect_episode(
            train_py_env, agent.collect_policy, rb_observer)

        # Use data from the buffer and update the agent's network.
        iterator = iter(replay_buffer.as_dataset(sample_batch_size=BATCH_SIZE))
        trajectories, _ = next(iterator)
        train_loss = agent.train(experience=trajectories)

        replay_buffer.clear()

        step = agent.train_step_counter.numpy()

        if step % LOG_INTERVAL == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % EVAL_INTERVAL == 0:
            avg_return = compute_avg_return(eval_env, agent.policy)
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
