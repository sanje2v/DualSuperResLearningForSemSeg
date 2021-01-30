from tqdm.auto import tqdm

from models import DSRL
from utils import *


def print_model(stage, dataset, **other_args):
    model = DSRL(stage, dataset['settings']).train()
    tqdm.write(str(model))
    info = "\nTotal training parameters: {0:,}\nTotal parameters: {1:,}".format(*countModelParams(model))
    tqdm.write(INFO(info))