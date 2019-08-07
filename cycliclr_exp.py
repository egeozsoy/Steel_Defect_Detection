from torch.optim.lr_scheduler import CyclicLR
import math


class CyclicExpLR(CyclicLR):

    def get_lr(self):
        """
        Different than the default pytorch version, this increases the lr rate exponentially(like fastai)
        """
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        pct = 1. + self.last_epoch / self.total_size - cycle
        if pct <= self.step_ratio:
            scale_factor = pct / self.step_ratio
        else:
            scale_factor = (pct - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            lr = base_lr * (max_lr / base_lr) ** pct
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs
