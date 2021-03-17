import numpy as np


class Accuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.dirty = False
        self.mean_accuracy = 0.0
        self.accuracies = []

    def update(self, pred, target, valid_labels_mask): # NOTE: This class is designed to calculate mIoU in batches of (pred, target) pairs
        assert pred.shape == target.shape, "BUG CHECK: 'pred' and 'target' must be of the same shape of (B, H, W)."
        assert len(pred.shape) == 3, "BUG CHECK: 'target' and 'pred' must be (B, H, W) channel-order dimensions."

        self.dirty = True

        pixels_correct = ((pred == target) * valid_labels_mask).sum()
        total_pixels = valid_labels_mask.sum()

        assert pixels_correct <= total_pixels, "BUG CHECK: 'pixels_correct' cannot be be greater than 'total_pixels'."

        self.accuracies.append(pixels_correct / total_pixels)

    def __call__(self):
        if self.dirty:
            self.dirty = False
            self.mean_accuracy = (np.mean(self.accuracies) * 100.)
        return self.mean_accuracy