# NOTE: This implementation has an accelerated version of hue jitter of TorchVision v0.8.1 implemention.
# This has been recorded to be 3 times faster. We achieve this by doing in-place hue rotation and parallelization.
# Other types of jitters were fast enough in their default implementation.
# Ref: https://stackoverflow.com/questions/8507885/shift-hue-of-an-rgb-color

import torch as t
import torchvision as tv
import torchvision.transforms.functional as F
import numpy as np
import numba as nb
import numbers
import random


class ColorJitter2(t.nn.Module):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @t.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    @nb.jit(nopython=True, parallel=True, cache=True, inline='always')
    def _accelerated_matmul(img, hue_rotation_matrix):
        for row in nb.prange(img.shape[-2]):
            for col in nb.prange(img.shape[-1]):
                img[0, row, col] = img[0, row, col] * hue_rotation_matrix[0, 0] + img[1, row, col] * hue_rotation_matrix[0, 1] + img[2, row, col] * hue_rotation_matrix[0, 2]
                img[1, row, col] = img[0, row, col] * hue_rotation_matrix[1, 0] + img[1, row, col] * hue_rotation_matrix[1, 1] + img[2, row, col] * hue_rotation_matrix[1, 2]
                img[2, row, col] = img[0, row, col] * hue_rotation_matrix[2, 0] + img[1, row, col] * hue_rotation_matrix[2, 1] + img[2, row, col] * hue_rotation_matrix[2, 2]

    @staticmethod
    def _rotate_hue(img, hue_rotation_matrix):
        hue_rotation_matrix = np.array(hue_rotation_matrix, dtype=img.numpy().dtype) # CAUTION: Important to keep data types to same type of float
        ColorJitter2._accelerated_matmul(img.numpy(), hue_rotation_matrix)
        return t.clip(img, min=0., max=1.)

    def forward(self, img):
        """
        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Color jittered image.
        """
        assert isinstance(img, t.Tensor), "BUG CHECK: Only 'torch.tensor' type of 'img' is supported."

        fn_idx = t.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = t.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = t.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = t.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = t.tensor(1.0).uniform_(hue[0], hue[1]).item()
                hue_factor_radians = hue_factor * 2.0 * np.pi

                # Prepare rotation matrix
                cosA = np.cos(hue_factor_radians)
                sinA = np.sin(hue_factor_radians)
                hue_rotation_matrix =\
                    [[cosA + (1.0 - cosA) / 3.0, 1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA, 1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA],
                     [1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA, cosA + 1./3.*(1.0 - cosA), 1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA],
                     [1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA, 1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA, cosA + 1./3. * (1.0 - cosA)]]
                img = ColorJitter2._rotate_hue(img, hue_rotation_matrix)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string