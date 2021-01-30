import torch as t
import torchvision as tv
from tqdm.auto import tqdm

from models import DSRL
import consts
from utils import *


def compile_model(weights, output_file, dataset, **other_args):
    # Compile given model using TorchScript and outputs a compiled file
    model = DSRL(stage=1, dataset_settings=dataset['settings']).eval()    # CAUTION: Important to set to eval mode

    # Load specified weights file
    model.load_state_dict(load_checkpoint_or_weights(weights)['model_state_dict'], strict=True)

    tqdm.write(INFO("Tracing model to compile..."))
    with t.jit.optimized_execution(True):
        trace = t.jit.trace(model, t.rand(1, consts.NUM_RGB_CHANNELS, *DSRL.MODEL_INPUT_SIZE))
    trace.save(output_file)
    tqdm.write(INFO("Compiled model saved to specified file."))