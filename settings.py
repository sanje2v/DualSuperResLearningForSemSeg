import os.path
import collections
import torch as t

DEFAULT_DTYPE = t.float32
t.set_default_dtype(DEFAULT_DTYPE)

version_tuple = collections.namedtuple("Row", ["major", "minor"])
MIN_PYTHON_VERSION = version_tuple(major=3, minor=7)
MIN_PYTORCH_VERSION = version_tuple(major=1, minor=7)

PROGRESSBAR_FORMAT = '{desc}: {percentage:.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}'
PARAMS_FILE = 'params.txt'
WEIGHTS_ROOT_DIR = 'weights'
WEIGHTS_DIR = os.path.join(WEIGHTS_ROOT_DIR, 'stage{stage}')
FINAL_WEIGHTS_FILE  = 'final.weights'
CHECKPOINTS_DIR = os.path.join(WEIGHTS_DIR, 'checkpoints')
CHECKPOINT_FILE = 'epoch{epoch}.checkpoint'
LOGS_DIR = os.path.join('logs', 'stage{stage}', '{mode}')
OUTPUTS_DIR = 'outputs'
PROFILING_FILE = 'profiling.json'
DATASETS_DIR = 'datasets'

CITYSCAPES_DATASET_DATA_DIR = os.path.join(DATASETS_DIR, 'Cityscapes', 'data')