import numpy as np


class Accuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mean_accuracy = []

    def update(self, pred, target, valid_labels_mask): # NOTE: This class is designed to calculate mIoU in batches of (pred, target) pairs
        assert pred.shape == target.shape, "BUG CHECK: 'pred' and 'target' must be of the same shape of (B, H, W)."
        assert len(pred.shape) == 3, "BUG CHECK: 'target' and 'pred' must be (B, H, W) channel-order dimensions."

        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        pixels_correct = np.sum((pred == target) * valid_labels_mask, axis=0, keepdims=True)
        total_pixels = np.sum(valid_labels_mask, axis=0, keepdims=True)

        self.mean_accuracy.append(np.mean(pixels_correct/total_pixels))

    def __call__(self):
        return np.mean(self.mean_accuracy)    # CAUTION: We use 'nanmean' to ignore any Nan values