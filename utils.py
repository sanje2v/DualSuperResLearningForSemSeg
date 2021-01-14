import os
import os.path
import platform
import ctypes
import termcolor
from tqdm.auto import tqdm as tqdm
from datetime import datetime, timedelta
import torch as t
import numpy as np

import consts


_starttimes_dict = {'default': datetime.now()}
def timeit(message=None, label='default'):
    global _starttimes_dict

    difftime = None
    now = datetime.now()
    if label in _starttimes_dict and message:
        difftime = now - _starttimes_dict[label]
        print("{0:s}: {1:.3f} secs".format(label, difftime.total_seconds()))
    _starttimes_dict[label] = now
    return difftime

def makeSecondsPretty(time_elasped):
    if time_elasped >= 86400.:
        time_elasped /= 86400.
        time_elasped_unit = 'days'
    elif time_elasped >= 3600.:
        time_elasped /= 3600.
        time_elasped_unit = 'hrs'
    elif time_elasped >= 60.:
        time_elasped /= 60.
        time_elasped_unit = 'mins'
    else:
        time_elasped_unit = 'secs'

    return "{0:.2f} {1:s}".format(time_elasped, time_elasped_unit)

class timethis:
    def __init__(self, message):
        self.message = message
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self):
        time_elasped = (datetime.now() - self. start_time).total_seconds()
        tqdm.write(message.format(makeSecondsPretty(time_elasped)))


def INFO(text):
    return termcolor.colored("INFO: {:}".format(text), 'green')

def CAUTION(text):
    return termcolor.colored("CAUTION: {:}".format(text), 'yellow')

def FATAL(text):
    return termcolor.colored("FATAL: {:}".format(text), 'red', attrs=['reverse', 'blink'])


def check_version(version, major, minor):
    if type(version) == str:
        version = tuple(int(x) for x in version.split('.'))

    return version[0] >= major and version[1] >= minor


def prevent_system_sleep():
    # NOTE: Only Windows OS supported for automatic system sleep disable until process ends function.
    #       For other OSes, the user should be advised to do so through their system settings.
    if platform.system() == 'Windows':
        # We prevent the system from sleeping but allow the display to turn off
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001

        return (ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) != 0)
    return False


def swapTupleValues(t):
    assert type(t) in [tuple, list] and len(t) == 2, "Only tuple of size 2 is supported!"
    return type(t)((t[1], t[0]))


def hasExtension(filename, extension):
    return os.path.splitext(filename)[-1].lower() == extension.lower()


def isCUDAdevice(device):
    return device.startswith(('gpu', 'cuda'))

def countNoOfModelParams(model):
    num_learning_parameters = num_total_parameters = 0
    for param in model.parameters():
        num_total_parameters += param.numel()
        if param.requires_grad:
            num_learning_parameters += param.numel()
    return num_learning_parameters, num_total_parameters

def load_checkpoint_or_weights(filename, map_location=t.device('cpu')):
    return t.load(filename, map_location)

def save_checkpoint(dir, filename, **checkpoint_vars):
    os.makedirs(dir, exist_ok=True)
    t.save(checkpoint_vars, os.path.join(dir, filename))

def save_weights(dir, filename, model):
    os.makedirs(dir, exist_ok=True)
    t.save({'model_state_dict': model.state_dict()}, os.path.join(dir, filename))

def make_output_visualization(SSSR_output, input_image, model_output_size, class_rgb_color, blend_factor=0.3):
    assert SSSR_output.shape == input_image.shape

    output_image = np.zeros((model_output_size[0], model_output_size[1] * 3, consts.NUM_RGB_CHANNELS), dtype=np.uint8)
    argmax_map = np.argmax(SSSR_output, axis=0)
    
    for y in range(model_output_size[0]):
        for x in range(model_output_size[1]):
            output_image[y, x, :] = input_image.getpixel((x, y))
            output_image[y, x + model_output_size[1], :] = class_rgb_color[(argmax_map[y, x])]

            blended_pixels = (1. - blend_factor) * output_image[y, x, :] + blend_factor * output_image[y, x + model_output_size[1], :]
            output_image[y, 2*x + model_output_size[1], :] = np.clip(blended_pixels, a_min=0, a_max=255).astype(np.uint8)
    return output_image