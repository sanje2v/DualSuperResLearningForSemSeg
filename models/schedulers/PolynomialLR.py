# NOTE: Source code adapted from public repo: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay
import math
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial
        last_epoch: To continue from an epoch else left to -1 to start from beginning
    """
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate, power, last_epoch=-1, verbose=False):
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def __calc_poly_decayed_lr(self, initial_lr):
        return (initial_lr - self.end_learning_rate)\
                * math.pow(1. - self.last_epoch / self.max_decay_steps, self.power)\
                + self.end_learning_rate

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use 'get_last_lr()'.", UserWarning)

        return [self.__calc_poly_decayed_lr(base_lr) for base_lr in self.base_lrs] \
                if self.last_epoch > 0 else self.base_lrs