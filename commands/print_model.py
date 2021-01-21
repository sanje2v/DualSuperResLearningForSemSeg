from tqdm.auto import tqdm

from models import DSRL
from utils import *


def print_model(stage, **other_args):
    model = DSRL(stage)
    tqdm.write(str(model))
    info = "\nTotal training parameters: {0:,}\nTotal parameters: {1:,}".format(*countNoOfModelParams(model))
    tqdm.write(INFO(info))