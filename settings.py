import os.path
import collections
import torch as t
import apex.parallel
import torchvision as tv
from functools import partial

from datasets import Cityscapes


# Minimum required library versions
version_tuple = collections.namedtuple("Row", ["major", "minor"])
MIN_PYTHON_VERSION = version_tuple(major=3, minor=7)
MIN_PYTORCH_VERSION = version_tuple(major=1, minor=7)
MIN_TORCHVISION_VERSION = version_tuple(major=0, minor=8)
MIN_NUMPY_VERSION = version_tuple(major=1, minor=19)

# PyTorch library
DEFAULT_DTYPE = t.float32
t.set_default_dtype(DEFAULT_DTYPE)
# NOTE: Put all batch normalization class types here so that 'freeze batch normalization layer' can work properly
BATCHNORM_MODULE_CLASSES = (t.nn.BatchNorm1d, t.nn.BatchNorm2d, t.nn.BatchNorm3d, t.nn.SyncBatchNorm, apex.parallel.SyncBatchNorm)
SUPPORTED_DEVICES = ['cpu', 'gpu']
SUPPORTED_DISTRIBUTED_BACKENDS = ['gloo', 'mpi', 'nccl']
RANDOM_SEED = 54321

# Apex multi precision library
AMP_OPTIMIZATION_OPTIONS = [None, 'O0', 'O1', 'O2', 'O3']

# Default values for commandline arguments
DEFAULT_DEVICE = 'gpu'
DEFAULT_AMP_OPTIMIZATION_OPTION = AMP_OPTIMIZATION_OPTIONS[0]
DEFAULT_NUM_WORKERS = 4
DEFAULT_VAL_INTERVAL = 10
DEFAULT_CHECKPOINT_INTERVAL = 5
DEFAULT_CHECKPOINT_HISTORY = 5
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_END_LEARNING_RATE = 0.001
DEFAULT_MOMENTUM = 0.9
DEFAULT_WEIGHTS_DECAY = 0.0005
DEFAULT_POLY_POWER = 0.9
DEFAULT_LOSS_WEIGHTS = [0.1, 1.0]


PROGRESSBAR_FORMAT = '{desc}: {percentage:.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}'
PARAMS_FILE = 'params.txt'
EXPERIMENTS_ROOT_DIR = 'experiments'
WEIGHTS_ROOT_DIR = 'weights'
WEIGHTS_DIR = os.path.join(WEIGHTS_ROOT_DIR, 'stage{stage}')
FINAL_WEIGHTS_FILE  = 'final.weights'
CHECKPOINTS_DIR = os.path.join(WEIGHTS_DIR, 'checkpoints')
CHECKPOINT_FILE = 'epoch{epoch}.checkpoint'
LOGS_DIR = os.path.join('logs', 'stage{stage}', '{mode}')
OUTPUTS_DIR = 'outputs'
PROFILING_FILE = 'profiling.json'
DATASETS_DIR = 'datasets'

DATASETS =\
{
    'cityscapes': {'path': os.path.join(DATASETS_DIR, 'Cityscapes', 'data'),
                   'splits': ['train', 'val', 'test'],
                   'class': partial(tv.datasets.Cityscapes,
                                    mode='fine',
                                    target_type='semantic'),
                   'settings': Cityscapes.settings},
}
DATASETS = {k.casefold(): v for k, v in DATASETS.items()}   # CAUTION: Make sure the dataset names (keys) and split values are all lowercase

VARIABLES_IN_CHECKPOINT =\
['device', 'mixed_precision', 'amp_state_dict', 'disable_cudnn_benchmark', 'num_workers', 'val_interval', 'checkpoint_interval', 'checkpoint_history', 'init_weights',
 'batch_size', 'epochs', 'learning_rate', 'end_learning_rate', 'momentum', 'weights_decay', 'poly_power', 'stage', 'w1', 'w2', 'freeze_batch_norm', 'experiment_id',
 'description', 'early_stopping', 'CE_train_avg_loss', 'MSE_train_avg_loss', 'FA_train_avg_loss', 'Avg_train_loss', 'CE_val_avg_loss', 'MSE_val_avg_loss',
 'FA_val_avg_loss', 'Avg_val_loss', 'epoch', 'best_validation_dict', 'model_state_dict', 'optimizer_state_dict', 'amp_state_dict']