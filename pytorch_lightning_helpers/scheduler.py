#!/usr/bin/env python3

import functools
import warnings

import pytorch_lightning as pl

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch

torch.backends.cudnn.benchmark = True
from itertools import accumulate

import pytorch_lightning as pl


class DynamicWarmerScheduler(pl.Callback):
    def __init__(
        self,
        attribute,
        monitor,
        increase_rate,
        decay_rate,
        threshold,
        init_value,
        start_step=0,
        max_value=None,
        mode="increase_when_above",
    ):
        super().__init__()
        self.attribute = attribute
        self.monitor = monitor
        self.start_step = start_step
        self.increase_rate = increase_rate
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.init_value = init_value
        self.max_value = max_value
        self.criterion = self.get_criterion(mode)

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        logs = trainer.callback_metrics
        new_val = self.get_value(
            self.rgetattr(pl_module, self.attribute),
            logs.get(self.monitor),
            trainer.global_step,
        )
        self.rsetattr(pl_module, self.attribute, new_val)
        pl_module.log(f"GradientReverse/{self.attribute}", new_val, on_step=True)

    def get_criterion(self, mode):
        if mode == "increase_when_above":
            return lambda val: val < self.threshold
        elif mode == "increase_when_below":
            return lambda val: val > self.threshold
        else:
            raise ValueError(
                f'{mode} is not a mode in ["increase_when_below", "increase_when_above"]'
            )

    def get_value(self, old_val, monitor_val, global_step):
        if global_step < self.start_step:
            return 0
        if monitor_val is None:
            return old_val
        if old_val == 0:
            return self.init_value
        if self.criterion(monitor_val):
            val = old_val * self.decay_rate
        else:
            val = old_val * self.increase_rate
        if self.max_value is not None:
            val = min(val, self.max_value)
        val = max(val, self.init_value)
        return val

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        val = callback_state.get(f"GradientReverse/{self.attribute}", self.init_value)[
            "val"
        ]
        setattr(pl_module, self.attribute, val)

    def on_save_checkpoint(
        self,
        trainer,
        pl_module,
        checkpoint,
    ):
        checkpoint[f"GradientReverse/{self.attribute}"] = {
            "val": self.rgetattr(pl_module, self.attribute)
        }
        return checkpoint

    def rsetattr(self, obj, attr, val):
        pre, _, post = attr.rpartition(".")
        return setattr(self.rgetattr(obj, pre) if pre else obj, post, val)

    def rgetattr(self, obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))


class SwitchDataLoaderScheduler(pl.Callback):
    def __init__(self, max_epoch_each_stage):
        super().__init__()
        self.max_epoch_each_stage = list(
            accumulate(max_epoch_each_stage)
        )  # [1,4,6] -> [1,5,11]

    def on_train_epoch_start(self, trainer, pl_module):
        # preparing next epoch's data
        for i, e in enumerate(self.max_epoch_each_stage):
            if trainer.current_epoch == e - 1:
                trainer.datamodule.set_phase(i + 1)
