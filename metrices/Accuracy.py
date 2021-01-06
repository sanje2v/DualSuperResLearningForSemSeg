import numpy as np


class Accuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mean_accuracy = []

    def update(self, pred, target, valid_labels_mask): # NOTE: This class is designed to calculate mIoU in batches of (pred, target) pairs
        assert pred.shape == target.shape, "BUG CHECK: 'pred' and 'target' must be of the same shape of (B, H, W)."
        assert len(pred.shape) == 3, "BUG CHECK: 'target' and 'pred' must be (B, H, W) channel-order dimensions."

        pixels_correct = ((pred == target) * valid_labels_mask).sum()
        total_pixels = valid_labels_mask.sum()

        assert pixels_correct <= total_pixels, "BUG CHECK: 'pixels_correct' cannot be be greater than 'total_pixels'."

        self.mean_accuracy.append(pixels_correct / total_pixels)

    def __call__(self):
        return np.mean(self.mean_accuracy)    # CAUTION: We use 'nanmean' to ignore any Nan values