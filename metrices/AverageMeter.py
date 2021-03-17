import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.dirty = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.dirty = True
        self.val = val
        self.sum += val * n
        self.count += n

    def __call__(self):
        if self.dirty:
            self.dirty = False
            with np.errstate(divide='ignore', invalid='ignore'): # NOTE: We ignore division by zero
                self.avg = self.sum / self.count
        return self.avg