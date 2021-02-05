import os.path
from tqdm.auto import tqdm

from models import DSRL
from utils import *


def prune_weights(src_weights, dest_weights, dataset, **other_args):
    with t.no_grad():
        # Purne weights not needed for inference
        model = DSRL(stage=1, dataset_settings=dataset['settings']).eval()

        # Load source weights file
        model.load_state_dict(load_checkpoint_or_weights(src_weights)['model_state_dict'], strict=True)

        save_weights(*os.path.split(dest_weights), model)
        print(INFO("Output weight saved in '{:s}'.".format(dest_weights)))