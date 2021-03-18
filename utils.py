import os
import os.path
import io
import sys
import argparse
import inspect
import platform
import ctypes
import termcolor
from tqdm.auto import tqdm
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
        print("{0:s}: {1:.3f} secs".format(label, difftime.total_seconds()))
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
        print(self.message.format(makeSecondsPretty(time_elasped)))

# Ref: https://stackoverflow.com/questions/36986929/redirect-print-command-in-python-script-through-tqdm-write
class OverridePrintWithTQDMWriteAndLog:
    def __init__(self, log_filename = None):
        self.logfile = open(log_filename, 'w+') if log_filename else None
        self.old_stdout = sys.stdout
        self.old_print = inspect.builtins.print

    def write(self, text):
        self.old_stdout.write(text)
        if self.logfile:
            self.logfile.write(text)

    def flush(self):
        self.old_stdout.flush()
        if self.logfile:
            self.logfile.flush()

    def __enter__(self):
        sys.stdout = self

        def new_print(*args, **kwargs):
            # if tqdm.write raises error, use builtin print
            try:
                tqdm.write(*args, **kwargs)
            except:
                self.old_print(*args, ** kwargs)

        inspect.builtins.print = new_print
        return self

    def __exit__(self, type, value, traceback):
        if self.logfile:
            self.logfile.close()
        sys.stdout = self.old_stdout
        inspect.builtins.print = self.old_print

class ConditionalContextManager:
    def __init__(self, expr_to_check, func_true, func_false=lambda: None):
        assert all(callable(f) for f in [func_true, func_false]), "BUG CHECK: Both 'func_true' and 'func_false' arguments must be 'Callable' type!"
        self.ctx = func_true() if expr_to_check else func_false()

    def __enter__(self):
        return self.ctx.__enter__() if hasattr(self.ctx, '__enter__') else self.ctx

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if hasattr(self.ctx, '__exit__'):
            self.ctx.__exit__(exc_type, exc_value, exc_traceback)

class ValidateDatasetNameAndSplit(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        datasets = self.const
        dataset, split = values[0].casefold(), values[1].casefold()

        if dataset not in datasets:
            raise ValueError("Unknown dataset! Supported datasets are: {:s}.".format(', '.join(datasets)))

        splits = datasets[dataset]['splits']
        if split not in splits:
            raise ValueError("Unknown dataset split! Supported splits are: {:s}.".format(', '.join(splits)))

        setattr(namespace, self.dest, values)

class ValidateDatasetNameSplitAndIndex(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        datasets = self.const
        dataset, split, starting_index = values[0].casefold(), values[1].casefold(), values[2]

        if dataset not in datasets:
            raise ValueError("Unknown dataset! Supported datasets are: {:s}.".format(', '.join(datasets)))

        splits = datasets[dataset]['splits']
        if split not in splits:
            raise ValueError("Unknown dataset split! Supported splits are: {:s}.".format(', '.join(splits)))

        if not starting_index.isnumeric():
            raise ValueError("Starting index must be an integer greater or equal to 0!")

        starting_index = int(starting_index)
        if starting_index < 0:
            raise ValueError("Starting index must be an integer greater or equal to 0!")

        setattr(namespace, self.dest, [dataset, split, starting_index])

class ValidateDistributedTrainingOptions(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        backends = self.const
        master_addr,\
        master_port,\
        nodes,\
        devices_per_node,\
        backend,\
        init_method,\
        node_id = values[0], values[1], values[2], values[3], values[4].casefold(), values[5].casefold(), values[6]

        if not master_port.isnumeric():
            raise ValueError("Master port must be a positive integer!")

        master_port = int(master_port)

        if not nodes.isnumeric():
            raise ValueError("Nodes must be a positive integer!")

        nodes = int(nodes)
        if nodes < 1:
            raise ValueError("Nodes must be greater than 0!")

        if not devices_per_node.isnumeric():
            raise ValueError("GPUs per node must be a positive integer!")

        devices_per_node = int(devices_per_node)
        if devices_per_node < 1:
            raise ValueError("GPUs per node must be greater than 0!")

        if backend not in backends:
            raise ValueError("Supported backends must be one of: [{:s}]!".format(', '.join(backends)))

        if not getattr(t.distributed, "is_{:s}_available".format(backend), lambda: False)():
            raise ValueError("Backend '{:s}' is not properly configured!".format(backend))

        if init_method == ' ':
            init_method = None

        if not node_id.isnumeric():
            raise ValueError("Node id must be an integer greater or equal to 0!")

        node_id = int(node_id)

        setattr(namespace, self.dest, [master_addr, master_port, nodes, devices_per_node, backend, init_method, node_id])


def INFO(text, prefix=''):
    return termcolor.colored("{0:}INFO: {1:}".format(prefix, text), 'green')

def CAUTION(text, prefix=''):
    return termcolor.colored("{0:}CAUTION: {1:}".format(prefix, text), 'yellow')

def FATAL(text, prefix=''):
    return termcolor.colored("{0:}FATAL: {1:}".format(prefix, text), 'red', attrs=['reverse', 'blink'])


def check_version(version, major, minor):
    if type(version) == str:
        version = tuple(int(x) for x in version.split('.'))

    return version[0] >= major and version[1] >= minor


INVALID_FILENAME_CHARS = ('<', '>', ':', '"', '/', '\\', '|', '?')
def isInvalidFilename(filename):
    for invalid_char in INVALID_FILENAME_CHARS:
        if invalid_char in filename:
            return True
    return False

def getFilesWithExtension(dir, extension_or_tuple, with_path=False):
    if not type(extension_or_tuple) is tuple:
        extension_or_tuple = (extension_or_tuple,)
    extension_or_tuple = tuple(x.casefold() for x in extension_or_tuple)
    return [(os.path.join(dir, f) if with_path else f) for f in os.listdir(dir) if f.casefold().endswith(extension_or_tuple)]

def hasExtension(filename, extension):
    return os.path.splitext(filename)[-1].casefold() == extension.casefold()


def prevent_system_sleep():
    # NOTE: This function only supports disabling system sleep (until process ends function) on Windows OS.
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


def convertIntIfNumeric(x):
    return int(x) if x.isnumeric() else x


def hasCaseInsensitive(x, items):
    for item in items:
        if x.casefold() == item.casefold():
            return True
    return False


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
    return device.casefold() == 'gpu'

def countModelParams(model):
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

def save_weights(dir, filename, model_state_dict, mixed_precision, amp_state_dict=None):
    os.makedirs(dir, exist_ok=True)
    if mixed_precision and amp_state_dict is None:
        amp_state_dict = apex.amp.state_dict()
    t.save({'model_state_dict': model_state_dict, 'mixed_precision': mixed_precision, 'amp_state_dict': amp_state_dict},
           os.path.join(dir, filename))

def make_input_output_visualization(input_image, output_map, class_rgb_color, blend_factor=0.4):
    assert input_image.shape[-2:] == output_map.shape[-2:]
    assert len(input_image.shape) == 3 and len(output_map.shape) == 2
    assert blend_factor > 0.0 and blend_factor < 1.0

    input_image = input_image.astype(np.uint8)
    output_map = output_map.astype(input_image.dtype)
    output_image = np.empty_like(input_image)
    overlayed_image = np.empty_like(output_image)
    class_rgb_color = dict((key, convertListToNumbaList(class_rgb_color[key], nb.intp)) for key, value in class_rgb_color.items())
    class_rgb_color = convertDictToNumbaDict(class_rgb_color, nb.intp, nb.types.ListType(nb.intp))

    @nb.jit(nopython=True, parallel=True, cache=True, inline='always')
    def _acceleratedTasks(input_image, output_image, output_map, overlayed_image, blend_factor, class_rgb_color, dtype):
        for channel in nb.prange(input_image.shape[0]):
            for y in nb.prange(input_image.shape[1]):
                for x in nb.prange(input_image.shape[2]):
                    output_image[channel, y, x] = class_rgb_color[output_map[y, x]][channel]
                    overlayed_image[channel, y, x] = dtype(min((1. - blend_factor) * input_image[channel, y, x] +\
                                                               blend_factor * output_image[channel, y, x], 255))
        return np.concatenate((input_image, output_image, overlayed_image), axis=2)
    return _acceleratedTasks(input_image, output_image, output_map, overlayed_image, blend_factor, class_rgb_color, input_image.dtype.type)