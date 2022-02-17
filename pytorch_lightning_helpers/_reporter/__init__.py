import pytorch_lightning as pl
import toolz
import torch


class Reporter(pl.Callback):
    def __init__(self):
        self.write_fns = {}
        self.trainer = None
        self.pl_module = None
        self.logging_disabled = False
        self.stage = None
        self.val_first_batch = False

    def on_fit_start(self, trainer, pl_module):
        self.trainer = trainer
        self.pl_module = pl_module

    def register(self, tag, fn):
        self.write_fns[tag] = fn

    def register_dict(self, fns):
        for tag, fn in fns:
            self.write_fns[tag] = fn

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
            self.pl_module.log(name, *args, **kwargs),
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
        if tag not in self.write_fns:
            self.pl_module.log_dict(kwargs_dict)
        elif (
            self.trainer.global_step % self.trainer.log_every_n_steps == 0
            or self.stage == "val"
        ):
            for name, kwargs in kwargs_dict.items():
                kwargs = toolz.valmap(clean_data_type, kwargs)
                self.log_media_to_wandb(name, tag=tag, **kwargs)

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
