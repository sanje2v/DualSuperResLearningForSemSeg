import os
import os.path
import platform
import ctypes
import termcolor
from tqdm.auto import tqdm as tqdm
from datetime import datetime
import torch as t
import numpy as np
import numba as nb



_starttimes_dict = {'default': datetime.now()}
def timeit(message=None, label='default'):
    global _starttimes_dict

    difftime = None
    now = datetime.now()
    if label in _starttimes_dict and message:
        difftime = now - _starttimes_dict[label]
        tqdm.write("{0:s}: {1:.3f} secs".format(label, difftime.total_seconds()))
    _starttimes_dict[label] = now
    return difftime

def makeSecondsPretty(time_elasped):
    SECS_BOUND_WITH_UNIT = [(86400., 'days'), (3600., 'hrs'), (60., 'mins')]

    time_elasped_unit = 'secs'  # Else
    for secs_bound, unit in SECS_BOUND_WITH_UNIT:
        if time_elasped >= secs_bound:
            time_elasped /= secs_bound
            time_elasped_unit = unit
            break
    return "{0:.2f} {1:s}".format(time_elasped, time_elasped_unit)

class timethis:
    def __init__(self, message):
        self.message = message
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        time_elasped = (datetime.now() - self.start_time).total_seconds()
        tqdm.write(self.message.format(makeSecondsPretty(time_elasped)))


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


def getFilesWithExtension(dir, extension_or_tuple, with_path=False):
    if not type(extension_or_tuple) is tuple:
        extension_or_tuple = (extension_or_tuple,)
    extension_or_tuple = tuple(x.lower() for x in extension_or_tuple)
    return [(os.path.join(dir, f) if with_path else f) for f in os.listdir(dir) if f.lower().endswith(extension_or_tuple)]


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


def convertDictToNumbaDict(py_dict, key_type, value_type):
    nb_dict = nb.typed.Dict.empty(key_type, value_type)

    for key, value in py_dict.items():
        nb_dict[key] = value
    return nb_dict

def convertListToNumbaList(py_list, item_type):
    nb_list = nb.typed.List.empty_list(item_type)

    for item in py_list:
        nb_list.append(item)
    return nb_list

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

def make_input_output_visualization(input_image, output_map, class_rgb_color, blend_factor=0.4):
    assert input_image.shape[-2:] == output_map.shape[-2:]
    assert len(input_image.shape) == 3 and len(output_map.shape) == 2
    assert blend_factor > 0.0 and blend_factor < 1.0

    input_image = input_image.astype(np.uint8)
    output_image = np.empty_like(input_image)
    overlayed_image = np.empty_like(output_image)
    arch_int_dtype = nb.intp    # Numba complains with warning if integer size is not same to architecture default int size
    class_rgb_color = dict((key, convertListToNumbaList(class_rgb_color[key], arch_int_dtype)) for key, value in class_rgb_color.items())
    class_rgb_color = convertDictToNumbaDict(class_rgb_color, arch_int_dtype, nb.types.ListType(arch_int_dtype))

    @nb.jit(nopython=True, parallel=True, cache=True, inline='always')
    def _acceleratedTasks(input_image, output_image, output_map, overlayed_image, blend_factor, class_rgb_color):
        for channel in nb.prange(input_image.shape[0]):
            for y in nb.prange(input_image.shape[1]):
                for x in nb.prange(input_image.shape[2]):
                    output_image[channel, y, x] = class_rgb_color[output_map[y, x]][channel]
                    overlayed_image[channel, y, x] = nb.uint8((1. - blend_factor) * input_image[channel, y, x] +\
                                                               blend_factor * output_image[channel, y, x] + 0.5)
        return np.concatenate((input_image, output_image, overlayed_image), axis=2)
    return _acceleratedTasks(input_image, output_image, output_map, overlayed_image, blend_factor, class_rgb_color)