# Reference: https://github.com/CYBORG-NIT-ROURKELA/Improving_Semantic_segmentation/blob/master/miou_calculation.py
import numpy as np


class mIoU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.dirty = False
        self.miou = 0.0
        self.ious = []

    def update(self, pred, target, valid_labels_mask): # NOTE: This class is designed to calculate mIoU in batches of (pred, target) pairs
        assert pred.shape == target.shape, "BUG CHECK: 'pred' and 'target' must be of the same shape of (B, H, W)."
        assert len(pred.shape) == 3, "BUG CHECK: 'target' and 'pred' must be (B, H, W) channel-order dimensions."

        self.dirty = True

        pred = pred + 1
        target = target + 1

        pred = pred * valid_labels_mask
        inter = pred * (pred == target)

        area_pred, _ = np.histogram(pred, bins=self.num_classes, range=(1, self.num_classes))
        area_inter, _ = np.histogram(inter, bins=self.num_classes, range=(1, self.num_classes))
        area_target, _ = np.histogram(target, bins=self.num_classes, range=(1, self.num_classes))
        area_union = area_pred + area_target - area_inter

        assert (area_inter <= area_union).all(), "BUG CHECK: Intersection area should always be less than or equal to union area."

        with np.errstate(divide='ignore', invalid='ignore'): # NOTE: We ignore division by zero
            self.ious.append(np.nanmean(area_inter / area_union))

    def __call__(self):
        if self.dirty:
            self.dirty = False
            self.miou = (np.nanmean(self.ious) * 100.)  # CAUTION: We use 'nanmean' to ignore any Nan values
        return self.miou