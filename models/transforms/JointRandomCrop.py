import torch as t
import torchvision.transforms.functional as F
import torch.nn.functional as F1
from PIL import Image
import numpy as np


class JointRandomCrop:
    def __init__(self, min_scale, max_scale):
        assert min_scale >= 1.0, "BUG CHECK: 'min_scale' must be greater than or equal to 1.0"
        assert min_scale < max_scale, "BUG CHECK: 'min_scale' must be greater than 'max_scale'"

        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, seg):
        assert (isinstance(img, x) and isinstance(seg, x) for x in [Image.Image, t.Tensor]), "BUG CHECK: 'img' and 'seg' must be of the same type"
        assert (isinstance(x, (Image.Image, t.Tensor)) for x in [img, seg]), "BUG CHECK: 'img' and 'seg' must be of either Image or Tensor type"
        assert (img.shape[-2:] == seg.shape[-2:]) if isinstance(img, t.Tensor) else (img.size[-2:] == seg.size[-2:]), "BUG CHECK: 'img' and 'seg' must be of same dimensions"

        # Get the default random number generator from numpy
        random_gen = np.random.default_rng()

        org_size = tuple(img.shape[-2:]) if isinstance(img, t.Tensor) else img.size    # CAUTION: For Image, size is (W, H) order in contrast to (H, W) for Tensor
        scale_factor = random_gen.uniform(self.min_scale, self.max_scale)

        if scale_factor > 1.0:
            # CAUTION: Interpolation mode must be 'nearest' for 'seg'
            if isinstance(img, Image.Image):
                crop_width = int(1.0 / scale_factor * org_size[0])
                crop_height = int(1.0 / scale_factor * org_size[1])
                crop_x = int(random_gen.uniform(0., (org_size[0] - crop_width) // 2))
                crop_y = int(random_gen.uniform(0., (org_size[1] - crop_height) // 2))
                crop_box = [crop_x,\
                            crop_y,\
                            crop_x+crop_width,\
                            crop_y+crop_height]

                img = img.resize(size=org_size, resample=Image.BILINEAR, box=crop_box)
                seg = seg.resize(size=org_size, resample=Image.NEAREST, box=crop_box)
            else:
                crop_width = int(1.0 /scale_factor * org_size[1])
                crop_height = int(1.0 /scale_factor * org_size[0])
                crop_box = [int(random_gen.uniform(0., (org_size[0] - crop_height) // 2)),\
                            int(random_gen.uniform(0., (org_size[1] - crop_width) // 2)),\
                            crop_height,\
                            crop_width]

                img = F.resized_crop(img, *crop_box, size=org_size, interpolation=Image.BILINEAR)
                seg = F.resized_crop(seg, *crop_box, size=org_size, interpolation=Image.NEAREST)

        return img, seg