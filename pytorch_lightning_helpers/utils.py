#!/usr/bin/env python3
from contextlib import nullcontext
import inspect
from collections.abc import Iterable
from functools import partial

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import _LRScheduler


def compose(*funcs):
    def f(**kwargs):
        ret = {}
        for f in funcs:
            ret |= supply_kwargs(f, kwargs|ret)
        return ret

    return f


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
            if key not in ["base_lrs"]
        }

    def load_state_dict(self, state_dict):
        # Override so that we can change the lr on resume
        self.__dict__.update({k: v for k, v in state_dict.items() if k != "base_lrs"})


def build_module_pipeline(model_cfg, optimizer_idx_map, train_stage="default"):
    def build_pipeline_item(
        module, fn, start_step=-1, cond=True, enabled_optim=None, freeze=False
    ):
        module_fn = fn_cache[module][fn]

        def pipeline_item(
            module_fn, start_step, cond, optimizer_idx=None, freeze=False, **kwargs
        ):
            if isinstance(start_step, Iterable):
                start_step = max(start_step)
            current_optim_name = (
                None if optimizer_idx is None else optimizer_idx_map[int(optimizer_idx)]
            )
            if (
                enabled_optim is not None
                and current_optim_name is not None
                and enabled_optim != current_optim_name
            ):
                return {}
            if kwargs.get("step", 0) < start_step:
                return {}
            if not cond:
                return {}
            args = inspect.getfullargspec(module_fn).args
            grad_context = torch.no_grad if freeze else nullcontext
            with grad_context():
                return supply_kwargs(module_fn, kwargs|{'optimizer_idx':optimizer_idx})

        return partial(
            pipeline_item,
            module_fn=module_fn,
            start_step=start_step,
            cond=cond,
            freeze=freeze,
        )

    module_cache = torch.nn.ModuleDict()
    fn_cache = {}
    buffers = {}
    for k, v in model_cfg.modules.items():
        module_cache.update(v["modules"])
        fn_cache[k] = v["methods"]
        buffers.update(v.get("buffers", {}))

    pipelines = {}
    for pipeline_name, pipeline_cfg in model_cfg.pipelines[train_stage].items():
        pipeline = []
        for pipeline_cfg_item in pipeline_cfg:
            pipeline.append(build_pipeline_item(**pipeline_cfg_item))
        pipelines[pipeline_name] = compose(*pipeline)

    inference_pipeline = []
    for inference_pipeline_cfg_item in model_cfg.inference_pipeline:
        inference_pipeline.append(build_pipeline_item(**inference_pipeline_cfg_item))

    pipelines["inference"] = compose(*inference_pipeline)

    param_group = {}
    if hasattr(model_cfg, "param_group"):
        for k, v in model_cfg.param_group.items():
            group_module_dict = torch.nn.ModuleDict()
            for name in v:
                group_module_dict.update(model_cfg.modules[name]["modules"])
            param_group[k] = group_module_dict.parameters()
        modules_in_param_group = set(
            sum(OmegaConf.to_container(model_cfg.param_group).values(), [])
        )
        non_moduleless_blocks = {
            k: v for k, v in model_cfg.modules.items() if len(v["modules"]) != 0
        }
        modules_without_param_group = (
            set(non_moduleless_blocks.keys()) - modules_in_param_group
        )
        for name in modules_without_param_group:
            logger.warning(f"{name} does not belong to any param groups.")
    else:
        param_group["default"] = module_cache.parameters()

    return module_cache, pipelines, param_group, buffers


def build_loss(loss_cfg, train_stage):
    def build_loss_item(loss_fn, scale=1):
        def loss(scale, **kwargs):
            return {k: v * scale for (k, v) in supply_kwargs(loss_fn, kwargs).items()}

        return partial(loss, scale=scale)

    loss_fn_cache = {}
    for k, v in loss_cfg.loss_functions.items():
        loss_fn_cache[k] = build_loss_item(**v)

    loss_sets = {}
    for loss_set_name, loss_set_list in loss_cfg.loss_sets[train_stage].items():
        loss_sets[loss_set_name] = compose(*[loss_fn_cache[x] for x in loss_set_list])
    return loss_sets


def detach_list(model_output):
    return [detach_any(v) for v in model_output]


def detach_values(model_output):
    return {k: detach_any(v) for k, v in model_output.items()}


def detach_any(item):
    if type(item) == torch.Tensor:
        return item.detach()
    elif type(item) == dict:
        return detach_values(item)
    elif type(item) == list:
        return detach_list(item)
    else:
        return item

def supply_kwargs(fn, kwargs):
    argspec = inspect.getfullargspec(fn)
    if argspec.varkw is None:
        kwargs = {k: v for k, v in kwargs.items() if k in argspec.args}
    return fn(**kwargs)
