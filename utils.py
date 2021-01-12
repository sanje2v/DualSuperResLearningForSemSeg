import os
import os.path
import platform
import ctypes
import termcolor
from datetime import datetime
import torch as t
import numpy as np
from datasets.Cityscapes import settings as cityscapes_settings


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

def countNoOfModelParams(model, only_training_params=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not only_training_params))

def write_params_file(filename, *list_params):
    with open(filename, mode='w') as params_file:
        for params_str in list_params:
            if params_str:
                params_file.write(params_str)
                params_file.write('\n')     # NOTE: '\n' here automatically converts it to newline for the current platform

def load_checkpoint_or_weights(filename, map_location=t.device('cpu')):
    return t.load(filename, map_location)

def save_checkpoint(dir, filename, **checkpoint_vars):
    os.makedirs(dir, exist_ok=True)
    t.save(checkpoint_vars, os.path.join(dir, filename))

def save_weights(dir, filename, model):
    os.makedirs(dir, exist_ok=True)
    t.save({'model_state_dict': model.state_dict()}, os.path.join(dir, filename))