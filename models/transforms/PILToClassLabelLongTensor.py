import torch as t
import numpy as np
from PIL import Image
from datasets.Cityscapes import settings as cityscapes_settings


class PILToClassLabelLongTensor:
    def __call__(self, pic):
        assert isinstance(pic, Image.Image), "'pic' parameter must be a PIL image!"

        mapping_func = np.vectorize(lambda x: np.int64(cityscapes_settings.LABEL_MAPPING_DICT[x]))

        return t.from_numpy(mapping_func(pic))