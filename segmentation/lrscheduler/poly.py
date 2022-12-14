# import mat
import warnings

from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLR(_LRScheduler):

    def __init__(self, optimizer, total_iters=5, power=1.0, min_lr=0.0, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [(group["lr"] - self.min_lr) * decay_factor + self.min_lr for group in self.optimizer.param_groups]
