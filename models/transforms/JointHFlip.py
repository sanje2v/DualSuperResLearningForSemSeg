import torch as t
import torchvision.transforms.functional as F


class JointHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, seg):
        do_flip = t.rand(1) < self.p
        return ((F.hflip(x) if do_flip else x) for x in (img, seg))