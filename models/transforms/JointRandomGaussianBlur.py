import torch as t
import torchvision as tv


class JointRandomGaussianBlur(t.nn.Module):
    def __init__(self, kernel_size, p):
        super().__init__()
        self.kernel_size = kernel_size
        self.p = p

    def forward(self, img, seg):
        do_flip = t.rand(1) < self.p
        return (tv.transforms.GaussianBlur(kernel_size=self.kernel_size)(img) if do_flip else img), seg

    def __repr__(self):
        return self.__class__.__name__ + '()'