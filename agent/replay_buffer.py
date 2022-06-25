import reverb

from tf_agents.specs import tensor_spec
from tf_agents.agents import TFAgent
from tf_agents.replay_buffers import reverb_replay_buffer, ReverbReplayBuffer, ReverbAddTrajectoryObserver
from tf_agents.replay_buffers import reverb_utils

from .constants import *


def replay_buffer_observer(agent: TFAgent) -> (ReverbReplayBuffer, ReverbAddTrajectoryObserver):
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec
    )
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature
    )

    table = reverb.Table(
        TABLE_NAME,
        max_size=REPLAY_BUFFER_MAX_LENGTH,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature
    )

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=TABLE_NAME,
        sequence_length=2,
        local_server=reverb_server
    )
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        TABLE_NAME,
        sequence_length=2)

    return replay_buffer, rb_observer
