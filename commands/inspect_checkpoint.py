from tqdm.auto import tqdm

from utils import *


def inspect_checkpoint(checkpoint, **other_args):
    checkpoint_dict = load_checkpoint_or_weights(checkpoint)

    def prettyDictToStr(dict_):
        output = []
        for key in dict_:
            if isinstance(dict_[key], dict):
                output.append("{0}: {1}".format(key, prettyDictToStr(dict_[key])))
            elif isinstance(dict_[key], (np.ndarray, t.Tensor, list)):
                output.append(key)
            else:
                if isinstance(dict_[key], str):
                    output.append("{0:s}: \"{1}\"".format(key, str(dict_[key])))
                else:
                    output.append("{0:s}: {1}".format(key, str(dict_[key])))

        return "{{{:s}}}".format(', '.join(output))

    tqdm.write(prettyDictToStr(checkpoint_dict))