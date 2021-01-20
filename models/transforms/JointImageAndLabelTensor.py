import torch as t
import torchvision as tv
from PIL import Image
import numpy as np
import numba as nb

from utils import *


class JointImageAndLabelTensor(t.nn.Module):
    @staticmethod
    @nb.jit(nopython=True, parallel=True, cache=True, inline='always')
    def _acceleratedRemap(seg, label_mapping_dict):
        for row in nb.prange(seg.shape[0]):
            for col in nb.prange(seg.shape[1]):
                seg[row, col] = label_mapping_dict[seg[row, col]]

    def _PILToClassLabelLongTensor(self, seg):
        # NOTE: We convert to numba supported 'numpy.narray' type
        seg = np.array(seg, dtype=np.int64) # NOTE: Pytorch's loss wants it to be of this 'long/int64' int type

        # NOTE: We use accelerated and parallized C++ code here to remap from dictionary as it
        #       can cut time required to more than half compared to native Python code.
        JointImageAndLabelTensor._acceleratedRemap(seg, self.label_mapping_dict)
        return t.from_numpy(seg)


    def __init__(self, label_mapping_dict):
        assert isinstance(label_mapping_dict, dict), "BUG CHECK: 'label_mapping_dict' should be a dict."
        super().__init__()

        # NOTE: Python 'dict' type is not supported by Numba v0.52 so we convert it to 'numba.typed.Dict' type
        self.label_mapping_dict = convertDictToNumbaDict(label_mapping_dict, nb.types.int64, nb.types.int64)

    def forward(self, img, seg):
        return tv.transforms.ToTensor()(img), self._PILToClassLabelLongTensor(seg)