import torch as t
import torchvision as tv
from tqdm.auto import tqdm

from models import DSRL
import consts
from utils import *


def compile_model(weights, output_file, **other_args):
    # Compile given model using TorchScript and outputs a compiled file
    model = DSRL(stage=1).cpu().eval()    # CAUTION: Important to set to eval mode

    # Load specified weights file
    model.load_state_dict(load_checkpoint_or_weights(weights)['model_state_dict'], strict=True)

    with t.jit.optimized_execution(True):
        tqdm.write(INFO("Tracing model to compile..."))
        # CAUTION: Make sure batch size is greater than 1 for BatchNorm to NOT complain
        trace = t.jit.trace(model, t.rand(2, consts.NUM_RGB_CHANNELS, *DSRL.MODEL_INPUT_SIZE))
        trace.save(output_file)
        tqdm.write(INFO("Compiled model saved to specified file."))