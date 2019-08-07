from .register import register
from torch.optim.lr_scheduler import _LRScheduler
import pdb

@register.attach('poly_lr')
class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = max((1 - self.last_epoch / float(self.max_iter)) ** self.gamma, 0)
        return [base_lr * factor for base_lr in self.base_lrs]


@register.attach('warm_up_lr')
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, multiplier=10, warmup_iters=100, last_epoch=-1, **poly_args):

        self.last_epoch = last_epoch
        self.warmup_iters = warmup_iters
        self.multiplier = multiplier
        super(WarmUpLR, self).__init__(optimizer, last_epoch)
        
        self.after_scheduler = PolynomialLR(optimizer, **poly_args)
        self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
        

    def get_lr(self):
        if self.last_epoch > self.warmup_iters:
            return self.after_scheduler.get_lr()
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_iters + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.last_epoch > self.warmup_iters:
            self.after_scheduler.step(self.last_epoch - self.warmup_iters)
        else:
            super(WarmUpLR, self).step(epoch)
