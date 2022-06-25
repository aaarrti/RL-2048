from os import environ


EPOCHS = environ.get('EPOCHS', default=500)

COLLECT_STEPS_PER_EPOCH = environ.get('COLLECT_STEPS_PER_EPOCH', default=10)

REPLAY_BUFFER_MAX_LENGTH = environ.get('REPLAY_BUFFER_MAX_LENGTH', default=10000000)

BATCH_SIZE = environ.get('BATCH_SIZE', default=128)

LEARNING_RATE = environ.get('LEARNING_RATE', default=1e-4)

NUM_EVAL_EPISODES = environ.get('NUM_EVAL_EPISODES', default=10)

FC_LAYERS_PARAMETERS = environ.get('FC_LAYERS_PARAMETERS', default=[50, 100, 300, 500, 300, 50])

TABLE_NAME = environ.get('TABLE_NAME', default='uniform_table')

CHECKPOINT_DIR = 'agent/checkpoints'
