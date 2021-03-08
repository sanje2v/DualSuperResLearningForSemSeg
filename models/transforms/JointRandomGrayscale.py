import torch as t
import torchvision as tv


class JointRandomGrayscale(t.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, img, seg):
        return tv.transforms.RandomGrayscale(p=self.p)(img), seg

    def __repr__(self):
        return self.__class__.__name__ + '()'