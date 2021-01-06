# NOTE: Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomRotation
import torch as t
import torchvision.transforms.functional as F
import numbers
from collections.abc import Sequence
from PIL import Image
import numpy as np


class JointRandomRotate(t.nn.Module):
    def __init__(self, degrees, fill=(None, None)):
        super().__init__()
        self.degrees = self._setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if fill is not None:
            self._check_sequence_input(fill, "fill", req_sizes=(2, ))

        self.fill = fill

    @staticmethod
    def get_params(degrees) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        # Get the default random number generator from numpy
        random_gen = np.random.default_rng()

        angle = random_gen.uniform(float(degrees[0]), float(degrees[1]))
        return angle


    def forward(self, img, seg):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.
            seg (PIL Image or Tensor): Seg to be rotated with.

        Returns:
            PIL Image or Tensor: Rotated image.
            PIL Image or Tensor: Rotated segmentation.
        """
        assert any(isinstance(img, x) and isinstance(seg, x) for x in [Image.Image, t.Tensor]), "BUG CHECK: 'img' and 'seg' must be of the same type."
        assert (isinstance(x, (Image.Image, t.Tensor)) for x in [img, seg]), "BUG CHECK: 'img' and 'seg' must be of either Image or Tensor type."
        assert (img.shape[-2:] == seg.shape[-2:]) if isinstance(img, t.Tensor) else (img.size[-2:] == seg.size[-2:]), "BUG CHECK: 'img' and 'seg' must be of same dimensions."

        angle = self.get_params(self.degrees)

        if isinstance(seg, t.Tensor):
            seg = t.unsqueeze(seg, dim=0)

        img, seg = F.rotate(img, angle, resample=Image.BILINEAR, expand=False, fill=self.fill[0]),\
                   F.rotate(seg, angle, resample=Image.NEAREST, expand=False, fill=self.fill[1])

        if isinstance(seg, t.Tensor):
            seg = t.squeeze(seg, dim=0)

        return img, seg

    def _check_sequence_input(self, x, name, req_sizes):
        msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
        if not isinstance(x, Sequence):
            raise TypeError("{} should be a sequence of length {}.".format(name, msg))
        if len(x) not in req_sizes:
            raise ValueError("{} should be sequence of length {}.".format(name, msg))


    def _setup_angle(self, x, name, req_sizes=(2, )):
        if isinstance(x, numbers.Number):
            if x < 0:
                raise ValueError("If {} is a single number, it must be positive.".format(name))
            x = [-x, x]
        else:
            self._check_sequence_input(x, name, req_sizes)

        return [float(d) for d in x]