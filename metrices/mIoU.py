# Reference: https://github.com/CYBORG-NIT-ROURKELA/Improving_Semantic_segmentation/blob/master/miou_calculation.py
import numpy as np


class mIoU:
    @staticmethod
    def _np_batch_bincount(arr, minlength):
        return np.apply_along_axis(lambda x: np.bincount(x, minlength=minlength), axis=1, arr=arr)


    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.miou = []

    def update(self, pred, target): # NOTE: This class is designed to calculate mIoU in batches of (pred, target) pairs
        assert pred.shape == target.shape, "BUG CHECK: 'pred' and 'target' must be of the same shape of (B, H, W)!"
        assert len(pred.shape) == 3, "BUG CHECK: 'target' and 'pred' must be (B, H, W) channel-order dimensions!"

        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

        # Bincount of class detections
        bincount_pred = mIoU._np_batch_bincount(pred, minlength=self.num_classes)
        bincount_target = mIoU._np_batch_bincount(target, minlength=self.num_classes)

        # Category matrix
        category_matrix = target * self.num_classes + pred
        bincount_category_matrix = mIoU._np_batch_bincount(category_matrix, minlength=(self.num_classes*self.num_classes))

        # Confusion matrix
        confusion_matrix = bincount_category_matrix.reshape((-1, self.num_classes, self.num_classes))

        intersection = np.diagonal(confusion_matrix, axis1=1, axis2=2)
        union = bincount_pred + bincount_target - intersection

        with np.errstate(divide='ignore', invalid='ignore'): # NOTE: We ignore division by zero
            self.miou.append(np.nanmean(intersection / union))

    def __call__(self):
        return np.nanmean(self.miou)    # CAUTION: We use 'nanmean' to ignore any Nan values