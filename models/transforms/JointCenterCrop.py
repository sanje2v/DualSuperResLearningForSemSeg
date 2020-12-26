import torch as t
import torchvision.transforms.functional as F
import torch.nn.functional as F1
from PIL import Image
import numpy as np


class JointCenterCrop:
    def __init__(self, min_scale, max_scale):
        assert min_scale < max_scale, "BUG CHECK: 'min_scale' must be greater than 'max_scale'"

        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, seg):
        assert img.shape[-2:] == seg.shape[-2:], "BUG CHECK: 'img' and 'seg' must be of same dimensions"

        org_size = list(img.shape[-2:])
        scale_factor = np.random.default_rng().uniform(self.min_scale, self.max_scale)

        if scale_factor > 1.0:
            img = F1.interpolate(t.unsqueeze(img, dim=0), scale_factor=scale_factor, mode='bilinear', align_corners=True, recompute_scale_factor=True)
            img = t.squeeze(F.center_crop(img, output_size=org_size), dim=0)

            seg = F.resize(t.unsqueeze(seg, dim=0), size=[int(scale_factor * x) for x in org_size], interpolation=Image.NEAREST)
            seg = t.squeeze(F.center_crop(seg, output_size=org_size), dim=0)

        return img, seg