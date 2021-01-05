import torch as t
import torchvision.transforms.functional as F


class JointHFlip(t.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, seg):
        do_flip = t.rand(1) < self.p
        return ((F.hflip(x) if do_flip else x) for x in (img, seg))