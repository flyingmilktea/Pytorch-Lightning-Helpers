import torch
import csv
import os
from loguru import logger
import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.logger import _name, _version
from collections import defaultdict

from FastSpeech.utils import rgetattr, rsetattr

class WarmerScheduler(Callback):
    def __init__(self, attribute, warmup_steps, start_step=0):
        super().__init__()
        self.attribute = attribute
        self.start_step = start_step
        self.warmup_steps = warmup_steps

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        new_val = self.get_value(trainer.global_step)
        setattr(pl_module, self.attribute, new_val)
        pl_module.log(f"warmer/{self.attribute}", new_val, on_step=True)

    def get_value(self, global_step):
        val = (global_step - self.start_step) / self.warmup_steps
        return min(max(val, 0), 1)


class DynamicWarmerScheduler(Callback):
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
        coeff=False,
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
        self.coeff = coeff
        self.mode = mode

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        logs = trainer.callback_metrics
        new_val = self.get_value(
            rgetattr(pl_module, self.attribute),
            logs.get(self.monitor),
            trainer.global_step,
            self.coeff,
        )
        rsetattr(pl_module, self.attribute, torch.Tensor([new_val]))
        pl_module.log(f"warmer/{self.attribute}", new_val, on_step=True)

    def criterion(self, val, threshold):
        if self.mode == "increase_when_above":
            return val < threshold
        elif self.mode == "increase_when_below":
            return val > threshold
        else:
            raise ValueError(
                f'{self.mode} is not a mode in ["increase_when_below", "increase_when_above"]'
            )

    def get_value(self, old_val, monitor_val, global_step, coeff):
        if global_step < self.start_step:
            return 0
        if monitor_val is None:
            return old_val
        if old_val == 0:
            return self.init_value
        if coeff:
            monitor_val = monitor_val / old_val.type_as(monitor_val)

        if self.criterion(monitor_val, self.threshold):
            val = old_val * self.decay_rate
        else:
            val = old_val * self.increase_rate
        if self.max_value is not None:
            val = min(val, self.max_value)
        val = max(val, self.init_value)
        return val
