import sys

from utils import *
import scripts


def message_script_not_found(*unused):
    print(FATAL("Cannot find any matching implementation of given script name under 'scripts' directory."))

if __name__ == '__main__':
    with OverridePrintWithTQDMWriteAndLog():    # All calls to 'print()' is to be redirected to 'tqdm.write()'
        if len(sys.argv) < 2 or sys.argv[1].casefold() in ['-h', '--help']:
            print("Run specified scripts under 'scripts' directory.")
            print("Usage: scripts.py <SCRIPT_NAME> <SCRIPT_PARAMS>")
            print("<SCRIPT_PARAMS> can be '-h' or '--help' to show arguments accepted by the script.")
            exit(0)

        script_to_call = getattr(scripts, sys.argv[1], message_script_not_found)
        script_to_call(sys.argv[2:])