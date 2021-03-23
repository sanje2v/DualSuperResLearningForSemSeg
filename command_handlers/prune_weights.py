import os.path
import apex
from tqdm.auto import tqdm
import torch as t

from models import DSRL
from utils import *


@t.no_grad()
def prune_weights(src_weights, dest_weights, dataset, **other_args):
    # Purne weights not needed for inference
    model = DSRL(stage=1, dataset_settings=dataset['settings']).eval()

    # Load source weights file
    src_weights_dict = load_checkpoint_or_weights(src_weights)
    model.load_state_dict(src_weights_dict['model_state_dict'], strict=True)

    save_weights(*os.path.split(dest_weights), model.state_dict(), src_weights_dict['mixed_precision'], src_weights_dict['amp_state_dict'])
    print(INFO("Output weight saved in '{:s}'.".format(dest_weights)))