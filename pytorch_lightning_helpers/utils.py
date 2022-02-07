#!/usr/bin/env python3

from torch.optim.lr_scheduler import _LRScheduler


def compose(*funcs):
    def f(**kwargs):
        ret = {}
        for f in funcs:
            ret |= f(**{**kwargs, **ret})
        return ret

    return f


def build_loss(loss_fn, scale=1, start_step=0):
    def loss(step=None, **kwargs):
        if step is not None and step < start_step:
            return {}
        return {k: v * scale for (k, v) in loss_fn(**kwargs).items()}

    return loss


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """

    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps**0.5 * min(
            last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]

    def state_dict(self):
        # Override so that we can change the lr on resume
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in ["optimizer", "base_lrs"]
        }

    def load_state_dict(self, state_dict):
        # Override so that we can change the lr on resume
        self.__dict__.update({k: v for k, v in state_dict.items() if k != "base_lrs"})
