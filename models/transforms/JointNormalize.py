import torch as t
import torchvision as tv


class JointNormalize(t.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, img, seg):
        return tv.transforms.Normalize(self.mean, self.std)(img), seg

    def __repr__(self):
        return self.__class__.__name__ + '()'