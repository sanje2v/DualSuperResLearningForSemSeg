import os.path
import apex
from tqdm.auto import tqdm

from models import DSRL
from utils import *


def prune_weights(src_weights, dest_weights, dataset, **other_args):
    with t.no_grad():
        # Purne weights not needed for inference
        model = DSRL(stage=1, dataset_settings=dataset['settings']).eval()

        # Load source weights file
        src_weights_dict = load_checkpoint_or_weights(src_weights)
        if src_weights_dict['mixed_precision']:
            model = apex.amp.initialize(model, opt_level=src_weights_dict['mixed_precision'])

        model.load_state_dict(src_weights_dict['model_state_dict'], strict=True)
        if src_weights_dict['mixed_precision']:
            amp.load_state_dict(src_weights_dict['amp_state_dict'])

        save_weights(*os.path.split(dest_weights), model.state_dict(), src_weights_dict['mixed_precision'])
        print(INFO("Output weight saved in '{:s}'.".format(dest_weights)))