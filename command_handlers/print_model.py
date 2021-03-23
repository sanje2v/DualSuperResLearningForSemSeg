from tqdm.auto import tqdm

from models import DSRL
from utils import *


def print_model(stage, dataset, **other_args):
    model = DSRL(stage, dataset['settings']).train()
    print(str(model))
    log_string = "Total training parameters: {0:,}\nTotal parameters: {1:,}".format(*countModelParams(model))
    print(INFO(log_string, prefix='\n'))