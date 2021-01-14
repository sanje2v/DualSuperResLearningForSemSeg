import os.path
from tqdm.auto import tqdm as tqdm
from pydoc import locate as str2type

from utils import *


def edit_checkpoint(checkpoint, key, value, typeof, **other_args):
    checkpoint_dict = load_checkpoint_or_weights(checkpoint)
    checkpoint_dict[key] = str2type(typeof)(value)
    save_checkpoint(*os.path.split(checkpoint), **checkpoint_dict)