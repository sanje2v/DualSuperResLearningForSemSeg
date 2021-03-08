import torch as t
import torchvision as tv
from PIL import Image
import numpy as np


class JointImageAndLabelTensor(t.nn.Module):
    @staticmethod
    def _PILToClassLabelLongTensor(seg, label_mapping_dict):
        assert isinstance(seg, Image.Image), "BUG CHECK: 'seg' parameter must be a PIL image."

        seg = np.array(seg, dtype=np.int64)
        for key, value in label_mapping_dict.items():
            seg[seg == key] = value

        return t.from_numpy(seg)


    def __init__(self, label_mapping_dict):
        assert isinstance(label_mapping_dict, dict), "BUG CHECK: 'label_mapping_dict' should be a dict."
        super().__init__()

        self.label_mapping_dict = label_mapping_dict

    def forward(self, img, seg):
        return tv.transforms.ToTensor()(img), \
               JointImageAndLabelTensor._PILToClassLabelLongTensor(seg, self.label_mapping_dict)

    def __repr__(self):
        return self.__class__.__name__ + '()'