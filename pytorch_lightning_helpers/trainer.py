import os

import hydra
import munch
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichModelSummary

from pytorch_lightning_helpers import reporter
from pytorch_lightning_helpers.utils import build_loss, compose
import traceback



class BaseLightningModule(pl.LightningModule):
    def __init__(self, modules, process=None, lossfuncs=None):
        super().__init__()
        if process is not None:
            self.process = compose(*process)
        if lossfuncs is not None:
            self.lossfuncs = lossfuncs
            self.loss_map = self.lossfuncs["order"]
            self.train_losses = {}
            for name, losses in self.lossfuncs["train"].items():
                self.train_losses[name] = compose(
                    *[build_loss(**loss) for loss in losses]
                )
            self.val_loss = compose(
                *[build_loss(**loss) for loss in self.lossfuncs["val"]]
            )

        for k, v in modules.items():
            setattr(self, k, v)
            setattr(v, 'module', lambda: self)

    def process(self, optimizer_idx, **kwargs):
        raise NotImplementedError(
            "process had to be either defined in custom lightning module or passed as a list in config."
        )

    def set_config(self, config):
        self.config = munch.munchify(config)

    def forward(self, srcs, refs):
        results_dict = self.model(srcs, refs)
        return results_dict["out"].squeeze().cpu().numpy()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        stage_name = self.loss_map[int(optimizer_idx)]

        model_output = self.process(**batch, optimizer_idx=optimizer_idx)
        if model_output is None:
            return None
        loss_dict = self.train_losses[stage_name](
            **(batch | model_output), step=self.global_step
        )
        if len(loss_dict) == 0:
            return None
        loss_dict["loss"] = sum(map(torch.mean, loss_dict.values()))
        reporter.report_dict(
            {f"train_{stage_name}/" + k: torch.mean(v) for k, v in loss_dict.items()}
        )
        loss_dict = {k: v.detach() if k != "loss" else v for k, v in loss_dict.items()}
        return loss_dict

    def validation_step(self, batch, batch_idx):
        model_output = self.process(**batch, optimizer_idx=None)
        loss_dict = self.val_loss(**(batch | model_output), step=self.global_step)

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
                model_inference_output = self.forward(first_data | model_output)
                self.log_eval(batch, model_output, model_inference_output)
            except RuntimeError as e:
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
        model = instantiate(cfg.model)
        model.set_config(cfg)
        trainer.fit(model, dm, ckpt_path=cfg.last_ckpt)
    else:
        model = hydra.utils.get_method(cfg.model["_target_"])
        model = model.load_from_checkpoint(
            cfg.last_ckpt,
            process=instantiate(cfg.process),
            lossfuncs=instantiate(cfg.losses),
            modules=instantiate(cfg.modules),
            strict=False,
        )
        model.set_config(cfg)
        trainer.fit(model, dm)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    main()
