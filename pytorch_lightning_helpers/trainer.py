import os
import traceback

import hydra
import munch
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichModelSummary

from pytorch_lightning_helpers import reporter
from pytorch_lightning_helpers.utils import build_loss, build_module_pipeline


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self, model=None, lossfuncs=None, optimizer_order=None, train_stage="default"
    ):
        super().__init__()
        self.optimizer_idx_map = optimizer_order
        if lossfuncs is not None:
            self.losses = build_loss(
                lossfuncs,
                train_stage,
            )
        model, self.pipelines, self.param_group = build_module_pipeline(
            model,
            self.optimizer_idx_map,
            train_stage,
        )

        for k, v in model.items():
            setattr(self, k, v)
            setattr(v, "module", lambda: self)

    def set_config(self, config):
        self.config = munch.munchify(config)

    def forward(self, srcs, refs):
        # TODO consider moving inference pipeline here
        results_dict = self.lightning_module(srcs, refs)
        return results_dict["out"].squeeze().cpu().numpy()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
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
        loss_dict["loss"] = sum(map(torch.mean, loss_dict.values()))
        reporter.report_dict(
            {f"train_{stage_name}/" + k: torch.mean(v) for k, v in loss_dict.items()}
        )
        loss_dict = {k: v.detach() if k != "loss" else v for k, v in loss_dict.items()}
        if not loss_dict['loss'].requires_grad:
            return None
        return loss_dict

    def validation_step(self, batch, batch_idx):
        model_output = self.pipelines[self.optimizer_idx_map[0]](
            **batch, step=self.global_step
        )
        if model_output is None:
            return None
        loss_dict = self.losses["val"](**(batch | model_output), step=self.global_step)

        if len(loss_dict) == 0:
            return None
        loss_dict["loss"] = sum(map(torch.mean, loss_dict.values()))
        reporter.report_dict(
            {"valid/" + k: torch.mean(v) for k, v in loss_dict.items()}
        )

        if hasattr(self, "log_eval") and batch_idx == 0:
            first_data = {
                k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            try:
                reporter.logging_disabled = True
                model_inference_output = self.forward(first_data | model_output)
                reporter.logging_disabled = False
                if model_inference_output is not None:
                    self.log_eval(batch, model_output, model_inference_output)
            except Exception as e:
                traceback.print_exc()
                logger.error(e)

        return loss_dict

    def configure_callbacks(self):
        return [RichModelSummary(max_depth=4)]


@hydra.main(config_path=os.getcwd() + "/configs", config_name="config")
def main(cfg: DictConfig):
    """
    os.symlink(
        os.path.abspath(".hydra/config.yaml"),
        os.path.join(wandb.run.dir, "hydra-config.yaml"),
    )
    wandb.save("hydra-config.yaml")
    """

    os.chdir(hydra.utils.get_original_cwd())
    pl.utilities.seed.seed_everything(42, workers=True)
    with torch.no_grad():
        dm = instantiate(cfg.dm)
        trainer = instantiate(cfg.trainer)
    trainer.callbacks.append(reporter)

    if cfg.load_optimizer or cfg.last_ckpt is None:
        lightning_module = instantiate(cfg.lightning_module)
        lightning_module.set_config(cfg)
        trainer.fit(lightning_module, dm, ckpt_path=cfg.last_ckpt)
    else:
        lightning_module = hydra.utils.get_method(cfg.lightning_module["_target_"])
        params = {
            k: instantiate(v) if type(v) != str else v
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
