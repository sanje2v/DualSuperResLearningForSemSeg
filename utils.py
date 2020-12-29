import os.path
import platform
import ctypes
import termcolor
from datasets.Cityscapes import settings as cityscapes_settings


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


def INFO(text):
    return termcolor.colored("INFO: " + text, 'green')

def CAUTION(text):
    return termcolor.colored("CAUTION: " + text, 'yellow')

def FATAL(text):
    return termcolor.colored("FATAL: " + text, 'red', attrs=['reverse', 'blink'])