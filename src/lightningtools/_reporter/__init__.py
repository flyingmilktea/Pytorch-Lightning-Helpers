import lightning as L
import toolz
import torch
from lightning.pytorch.utilities import rank_zero_only


class Reporter(L.Callback):
    def __init__(self):
        self.write_fns = {}
        self.trainer = None
        self.pl_module = None
        self.logging_disabled = False
        self.stage = None
        self.val_first_batch = False
        self.delayed_report_storage = {}

    def on_fit_start(self, trainer, pl_module):
        self.trainer = trainer
        self.pl_module = pl_module

    def register(self, tag, fn):
        self.write_fns[tag] = fn

    def register_dict(self, fns):
        for tag, fn in fns:
            self.write_fns[tag] = fn

    @torch.no_grad()
    def delayed_report(self, name, tag=None, reducer=None, **kwargs):
        if self.pl_module is None:
            return
        if self.logging_disabled:
            return
        if not (
            self.trainer.global_step % self.trainer.log_every_n_steps == 0
            or self.stage == "val"
        ):
            return

        if name in self.delayed_report_storage:
            assert self.delayed_report_storage[name]["tag"] == tag
            if reducer is not None:
                reduced_kwargs = {}
                old_kwargs = self.delayed_report_storage[name]
                for k, v in kwargs.items():
                    if k not in old_kwargs or k not in reducer:
                        reduced_kwargs[k] = v
                    else:
                        reduced_kwargs[k] = reducer[k](old_kwargs[k], v)
                self.delayed_report_storage[name] = old_kwargs | reduced_kwargs
            else:
                self.delayed_report_storage[name] |= kwargs
        else:
            self.delayed_report_storage[name] = {"tag": tag, **kwargs}

    @torch.no_grad()
    def flush_delayed_report(self):
        for name, storage in self.delayed_report_storage.items():
            self.report(name=name, **storage)
        self.delayed_report_storage.clear()

    @torch.no_grad()
    def report(self, name, *args, tag=None, **kwargs):
        # TODO: solve logged multiple times for multiple optimizer_idx
        if self.pl_module is None:
            return
        if self.logging_disabled:
            return
        if tag not in self.write_fns:
            args = list(recursive_map(clean_data_type, args))
            kwargs = recursive_valmap(clean_data_type, kwargs)
            self.pl_module.log(name, *args, **kwargs, sync_dist=True)
        elif (
            self.trainer.global_step % self.trainer.log_every_n_steps == 0
            or self.stage == "val"
        ):
            args = list(recursive_map(clean_data_type, args))
            kwargs = recursive_valmap(clean_data_type, kwargs)
            self.log_media_to_wandb(name, *args, tag=tag, **kwargs)

    @torch.no_grad()
    def report_dict(self, kwargs_dict, *, tag=None):
        if self.pl_module is None:
            return
        if self.logging_disabled:
            return
        if not self.trainer.is_global_zero:
            return
        if tag not in self.write_fns:
            self.pl_module.log_dict(kwargs_dict, sync_dist=True)
        elif (
            self.trainer.global_step % self.trainer.log_every_n_steps == 0
            or self.stage == "val"
        ):
            for name, kwargs in kwargs_dict.items():
                kwargs = toolz.valmap(clean_data_type, kwargs)
                self.log_media_to_wandb(name, tag=tag, **kwargs)

    @rank_zero_only
    def log_media_to_wandb(self, name, *args, tag, **kwargs):
        if self.stage == "val":
            name = f"val_{name}"
        self.pl_module.logger.experiment.log(
            {name: self.write_fns[tag](*args, **kwargs)},
            step=self.trainer.global_step,
            commit=False,
        )

    def on_sanity_check_start(self, *args, **kwargs):
        self.logging_disabled = True

    def on_sanity_check_end(self, *args, **kwargs):
        self.logging_disabled = False

    def on_validation_batch_start(self, batch_idx, *args, **kwargs):
        if self.val_first_batch:
            self.logging_disabled = False
            self.val_first_batch = False
        else:
            self.logging_disabled = True

    def on_validation_epoch_start(self, *args, **kwargs):
        self.stage = "val"
        self.val_first_batch = True

    def on_train_batch_start(self, *args, **kwargs):
        self.stage = "train"
        self.logging_disabled = False

    def on_train_batch_end(self, *args, **kwargs):
        self.flush_delayed_report()

    def on_validation_batch_end(self, *args, **kwargs):
        self.flush_delayed_report()


def clean_data_type(data):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu()
    return data


def recursive_map(func, seq):
    for item in seq:
        if type(item) in [list, tuple]:
            yield type(item)(recursive_map(func, item))
        elif type(item) in [dict]:
            yield type(item)(recursive_valmap(func, item))
        else:
            yield func(item)


def recursive_valmap(func, seq):
    ret = {}
    for k, item in seq.items():
        if type(item) in [list, tuple]:
            ret[k] = type(item)(recursive_map(func, item))
        elif type(item) in [dict]:
            ret[k] = type(item)(recursive_valmap(func, item))
        else:
            ret[k] = func(item)
    return ret
