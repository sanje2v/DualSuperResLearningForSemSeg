import os.path
import collections
import torch as t

DEFAULT_DTYPE = t.float32
t.set_default_dtype(DEFAULT_DTYPE)

version_tuple = collections.namedtuple("Row", ["major", "minor"])
MIN_PYTHON_VERSION = version_tuple(major=3, minor=7)
MIN_PYTORCH_VERSION = version_tuple(major=1, minor=7)
MIN_TORCHVISION_VERSION = version_tuple(major=0, minor=8)
MIN_NUMPY_VERSION = version_tuple(major=1, minor=19)

STAGES = [1, 2, 3]
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
VARIABLES_IN_CHECKPOINT =\
            ['device', 'disable_cudnn_benchmark', 'num_workers', 'val_interval', 'checkpoint_interval', 'checkpoint_history',
             'init_weights', 'batch_size', 'epochs', 'learning_rate', 'end_learning_rate', 'momentum', 'weights_decay',
             'poly_power', 'stage', 'w1', 'w2', 'freeze_batch_norm', 'description', 'best_validation_dict',
             'CE_train_avg_loss', 'MSE_train_avg_loss', 'FA_train_avg_loss', 'Avg_train_loss', 'CE_val_avg_loss', 'MSE_val_avg_loss',
             'FA_val_avg_loss', 'Avg_val_loss']

CITYSCAPES_DATASET_DATA_DIR = os.path.join(DATASETS_DIR, 'Cityscapes', 'data')