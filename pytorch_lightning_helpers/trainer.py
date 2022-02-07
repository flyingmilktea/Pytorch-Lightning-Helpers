import argparse

import munch
import pytorch_lightning as pl
import torch
from hyperpyyaml import load_hyperpyyaml
from pytorch_lightning.callbacks import RichModelSummary

from pytorch_lightning_helpers import reporter
from pytorch_lightning_helpers.utils import build_loss, compose


class BaseLightningModule(pl.LightningModule):
    def __init__(self, modules, process=None, lossfuncs = None):
        super().__init__()
        if process is not None:
            self.process = compose(*process)
        if lossfuncs is not None:
            self.lossfuncs = lossfuncs
            self.loss_map = self.lossfuncs["order"]
            self.train_losses = {}
            for name, losses in self.lossfuncs["train"].items():
                self.train_losses[name] = compose(*[build_loss(**loss) for loss in losses])
            self.val_loss = compose(*[build_loss(**loss) for loss in self.lossfuncs["val"]])

        for k, v in modules.items():
            setattr(self, k, v)

    def process(self, optimizer_idx, **kwargs):
        raise NotImplementedError(
            "process had to be either defined in custom lightning module or passed as a list in config."
        )

    def set_config(self, config):
        self.config = munch.munchify(config)

    def forward(self, srcs, refs):
        results_dict = self.model(srcs, refs)
        return results_dict["out"].squeeze().cpu().numpy()

    def training_step(self, batch, batch_idx, optimizer_idx):
        stage_name = self.loss_map[optimizer_idx]

        model_output = self.process(**batch, optimizer_idx=optimizer_idx)
        loss_dict = self.train_losses[stage_name](
            **{**batch, **model_output}, step=self.global_step
        )
        if len(loss_dict) == 0:
            return None
        loss_dict["loss"] = sum(loss_dict.values())
        reporter.report_dict({"train/" + k: v for k, v in loss_dict.items()})
        loss_dict = {k: v.detach() if k != "loss" else v for k, v in loss_dict.items()}
        return loss_dict

    def validation_step(self, batch, batch_idx):
        model_output = self.process(**batch, optimizer_idx=None)
        loss_dict = self.val_loss(**{**batch, **model_output}, step=self.global_step)

        if len(loss_dict) == 0:
            return None
        loss_dict["loss"] = sum(loss_dict.values())
        reporter.report_dict({"valid/" + k: v for k, v in loss_dict.items()})
        return loss_dict

    def configure_callbacks(self):
        return [RichModelSummary(max_depth=3)]


def main(config_file, name=None):
    with torch.no_grad():
        with open(config_file) as f:
            loaded_yaml = load_hyperpyyaml(f)
    dm = loaded_yaml["dm"]
    trainer = loaded_yaml["trainer"]
    model = loaded_yaml["model"]
    model.set_config(loaded_yaml)
    trainer.callbacks.append(reporter)

    if loaded_yaml["load_optimizer"]:
        trainer.fit(model, dm, ckpt_path=loaded_yaml["last_ckpt"])
    else:
        model.load_from_checkpoint(
            loaded_yaml["last_ckpt"],
            process=loaded_yaml["process"],
            lossfuncs=loaded_yaml["losses"],
            modules=loaded_yaml["modules"],
            strict=False,
        )
        trainer.fit(model, dm)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--name", "-n", default=None, type=str)
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(**parse_args())
