import argparse

import pytorch_lightning as pl
from hyperpyyaml import load_hyperpyyaml
from toolz import curry

from pytorch_lightning_helpers.utils import scale_loss, compose
import munch
from loguru import logger


class BaseLightningModule(pl.LightningModule):
    def __init__(self, process, lossfuncs, modules):
        super().__init__()
        self.process = compose(*process)
        #TODO set self.model for forward

        self.lossfuncs = lossfuncs
        self.loss_map = self.lossfuncs["order"]
        self.train_losses = {}
        for name, losses in self.lossfuncs["train"].items():
            self.train_losses[name] = compose(*[scale_loss(**loss) for loss in losses])
        self.val_loss = compose(*[scale_loss(**loss) for loss in self.lossfuncs["val"]])

        for k, v in modules.items():
            setattr(self, k, v)

    def set_config(self, config):
        self.config = munch.munchify(config)

    def forward(self, srcs, refs):
        results_dict = self.model(srcs, refs)
        return results_dict["out"].squeeze().cpu().numpy()

    def training_step(self, batch, batch_idx, optimizer_idx):
        model_output = self.process(**batch, optimizer_idx=optimizer_idx)

        stage_name = self.loss_map[optimizer_idx]
        loss_dict = self.train_losses[stage_name](**model_output, **batch)
        totalloss = sum(loss_dict.values())
        loss_dict["loss"] = totalloss
        loss_dict[f"loss_{stage_name}"] = totalloss
        # TODO Needed?
        # loss_dict.update(media_dict)

        # log postprocessing -> new logger
        """
        for key, value in model_output.items():
            if "_audio" in key or "_graph" in key:
                media_dict[key] = value
        #media_dict.update(resize_mel(**model_output))
        if optimizer_idx == 0:
            self.log("train/total_generator_loss", totalloss)
        elif optimizer_idx == 1:
            self.log("train/total_discriminator_loss", totalloss)
        for k, v in loss_dict.items():
            if "loss_" in k:
                self.log(f"train/{k}", v)
        """
        return loss_dict

    def validation_step(self, batch, batch_idx):
        model_output = self.process(**batch)
        loss_dict = self.val_loss(**model_output, **batch)
        for k, v in loss_dict.items():
            if "loss_" in k:
                self.log("valid/{k}", v)
        return loss_dict


def main(config_file, name=None):
    overrides = {}
    if name is not None:
        overrides["name"] = name
    with open(config_file) as f:
        loaded_yaml = load_hyperpyyaml(f, overrides)

    dm = loaded_yaml["dm"]
    trainer = loaded_yaml["trainer"]
    model = loaded_yaml["model"]
    model.set_config(loaded_yaml)

    if loaded_yaml["load_optimizer"]:
        trainer.fit(model, dm, ckpt_path=loaded_yaml["last_ckpt"])
    else:
        model.load_from_checkpoint(
            loaded_yaml["last_ckpt"],
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
