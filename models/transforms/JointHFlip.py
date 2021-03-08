import torch as t
import torchvision.transforms.functional as F


class JointHFlip(t.nn.Module):
    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, img, seg):
        do_flip = t.rand(1) < self.p
        return ((F.hflip(x) if do_flip else x) for x in (img, seg))

    def __repr__(self):
        return self.__class__.__name__ + '()'