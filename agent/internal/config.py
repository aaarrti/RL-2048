num_iterations = 20000

initial_collect_steps = 100
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000

batch_size = 64
learning_rate = 1e-3
log_interval = 1

num_eval_episodes = 200
eval_interval = 1000

fc_layer_params = (100, 50)

env_provisioner_uri = 'host.docker.internal:50055'

game_base_uri = 'host.docker.internal:'
