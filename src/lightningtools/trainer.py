import logging
import os
import traceback
import warnings
from pathlib import Path

import hydra
import lightning as L
import munch
import ray
import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import RichModelSummary
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from lightningtools import reporter
from lightningtools.utils import build_loss, build_module_pipeline, detach_any

ray.data.DataContext.get_current().enable_progress_bars
ray_logger = logging.getLogger("ray")
while ray_logger.hasHandlers():
    ray_logger.removeHandler(ray_logger.handlers[0])


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        model=None,
        lossfuncs=None,
        optimizer_order=None,
        train_stage="default",
        gradient_clip_val=None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.optimizer_idx_map = optimizer_order
        self.gradient_clip_val = gradient_clip_val
        if lossfuncs is not None:
            self.losses = build_loss(
                lossfuncs,
                train_stage,
            )
        model, self.pipelines, self.param_group, buffers = build_module_pipeline(
            model,
            self.optimizer_idx_map,
            train_stage,
        )

        for k, v in model.items():
            setattr(self, k, v)
            setattr(v, "module", lambda: self)

        for k, v in buffers.items():
            if torch.is_tensor(v):
                self.register_buffer(k, v, persistent=False)
            else:
                setattr(self, k, v)

    def set_config(self, config):
        self.config = munch.munchify(config)

    def forward(self, srcs, refs):
        # TODO consider moving inference pipeline here
        results_dict = self.lightning_module(srcs, refs)
        return results_dict["out"].squeeze().cpu().numpy()

    def training_step(self, batch, batch_idx):
        optimizer_idx = batch_idx % len(self.optimizer_idx_map)
        stage_name = self.optimizer_idx_map[int(optimizer_idx)]
        if stage_name not in self.pipelines:
            return None
        model_output = self.pipelines[stage_name](
            **batch, optimizer_idx=optimizer_idx, step=self.global_step
        )
        if model_output is None:
            return None
        loss_dict = self.losses[stage_name](
            **(batch | model_output), step=self.global_step
        )
        if len(loss_dict) == 0:
            return None
        total_loss = sum(map(torch.mean, loss_dict.values()))

        if len(self.optimizer_idx_map) == 1:
            opt = self.optimizers()
        else:
            opt = self.optimizers()[optimizer_idx]
        opt.zero_grad()

        self.manual_backward(total_loss)
        self.clip_gradients(
            opt,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )
        opt.step()

        loss_dict["loss"] = total_loss
        if not total_loss.requires_grad:
            total_loss = None
        reporter.report_dict(
            {f"train_{stage_name}/" + k: torch.mean(v) for k, v in loss_dict.items()}
        )
        loss_dict = {
            k: detach_any(v) if k != "loss" else v for k, v in loss_dict.items()
        }

        model_output = {k: detach_any(v) for k, v in model_output.items()}
        return {
            "loss_dict": loss_dict,
            "model_output": model_output,
            "loss": total_loss,
        }

    def validation_step(self, batch, batch_idx):
        model_output = self.pipelines[self.optimizer_idx_map[0]](
            **batch, step=self.global_step
        )
        if model_output is None:
            return None
        loss_dict = self.losses["val"](**(batch | model_output), step=self.global_step)
        if len(loss_dict) == 0:
            return None
        total_loss = sum(map(torch.mean, loss_dict.values()))
        loss_dict["loss"] = total_loss
        reporter.report_dict(
            {"valid/" + k: torch.mean(v) for k, v in loss_dict.items()}
        )

        if hasattr(self, "log_eval") and batch_idx == 0:
            first_data = {
                k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            try:
                reporter.logging_disabled = True
                model_inference_output = self.forward(first_data)
                reporter.logging_disabled = False
                if model_inference_output is not None:
                    self.log_eval(batch, model_output, model_inference_output)
            except Exception as e:
                traceback.print_exc()
                logger.error(e)

        return {
            "loss_dict": loss_dict,
            "model_output": model_output,
            "loss": total_loss,
        }

    def configure_callbacks(self):
        return [RichModelSummary(max_depth=4)]


os.environ["HYDRA_MAIN_MODULE"] = "__main__"


@hydra.main(config_path=str(Path.cwd()) + "/configs", config_name="config", version_base='1.3')
def main(cfg: DictConfig):
    """
    os.symlink(
        os.path.abspath(".hydra/config.yaml"),
        os.path.join(wandb.run.dir, "hydra-config.yaml"),
    )
    wandb.save("hydra-config.yaml")
    """
    os.chdir(hydra.utils.get_original_cwd())

    def logging_setup_func():
        # disable internal warnings from ray data
        warnings.filterwarnings(action="ignore")

    ray.init(
        runtime_env={"worker_process_setup_hook": logging_setup_func},
        log_to_driver=False,
    )

    L.fabric.utilities.seed.seed_everything(42, workers=True)
    with torch.no_grad():
        dm = instantiate(cfg.data_module.data_module)
        trainer = instantiate(cfg.trainer)
    trainer.callbacks.append(reporter)

    if cfg.load_optimizer or cfg.last_ckpt is None:
        lightning_module = instantiate(cfg.lightning_module)
        lightning_module.set_config(cfg)
        trainer.fit(lightning_module, dm, ckpt_path=cfg.last_ckpt)
    else:
        lightning_module = hydra.utils.get_method(cfg.lightning_module["_target_"])
        params = {
            k: instantiate(v) if isinstance(v, str) else v
            for k, v in cfg.lightning_module.items()
            if k != "_target_"
        }

        lightning_module = lightning_module.load_from_checkpoint(
            cfg.last_ckpt,
            **params,
            strict=False,
        )
        lightning_module.set_config(cfg)
        trainer.fit(lightning_module, dm)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
