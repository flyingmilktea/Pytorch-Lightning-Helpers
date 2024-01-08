import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

from lightningtools.scheduler import (  # noqa: F401 isort:skip # pylint:disable=unused-import
    WarmerScheduler,
    DynamicWarmerScheduler,
    SwitchDataLoaderScheduler,
)


class OutlierDetector(Callback):
    def __init__(
        self,
        dirpath,
        log_key,
        start_log=10000,
        stats_momentum=0.992,
        score_momentum=0.5,
        sd_threshold=4,
    ):
        super().__init__()
        self.loss_mean_logs = defaultdict(lambda: 0)
        self.loss_sd_logs = defaultdict(lambda: 1)
        self.outlier_logs = defaultdict(dict)
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(exist_ok=True, parents=True)
        self.log_key = log_key
        self.start_log = start_log
        self.score_momentum = score_momentum
        self.stats_momentum = stats_momentum
        self.sd_threshold = sd_threshold

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        if len(outputs) == 0:
            return
        outputs = outputs["loss_dict"]
        self.find_and_log_outliers(trainer, outputs, batch)

    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, *args, **kwargs
    ):
        outputs = outputs["loss_dict"]
        self.find_and_log_outliers(trainer, outputs, batch)

    def find_and_log_outliers(self, trainer, outputs, batch):
        for loss_name, loss in outputs.items():
            if len(loss.shape) == 0 or len(loss) == 1:
                continue
            if trainer.global_step > self.start_log:
                z_score = self.get_z_score(loss_name, loss)
                self.log_outliers(loss_name, loss, z_score, batch)
            self.update_stats(loss_name, loss)

    def on_validation_epoch_end(self, *args, **kwargs):
        self.flush_logs()

    def get_z_score(self, loss_name, loss):
        z_score = (loss - self.loss_mean_logs[loss_name]) / self.loss_sd_logs[loss_name]
        return z_score

    def log_outliers(self, loss_name, loss, z_score, batch):
        for i, data_key in enumerate(batch[self.log_key]):
            if not z_score[i] > self.sd_threshold:
                continue
            outlier_record = self.get_record(loss_name, z_score[i], data_key)
            self.outlier_logs[loss_name][data_key] = outlier_record

    def get_record(self, loss_name, z_score, data_key):
        if data_key in self.outlier_logs[loss_name]:
            prev_record = self.outlier_logs[loss_name][data_key]
            count = prev_record["count"] + 1
            prev_z_score = prev_record["average_z_score"]
            average_z_score = self.moving_average(
                prev_z_score, z_score, self.score_momentum
            )
        else:
            count = 1
            average_z_score = z_score
        return {
            "name": data_key,
            "outlier_score": np.log(count + 1) * average_z_score.item(),
            "average_z_score": average_z_score.item(),
            "count": count,
        }

    def moving_average(self, old_val, new_val, momentum):
        return old_val * momentum + new_val * (1 - momentum)

    def update_stats(self, loss_name, loss):
        sd, mean = torch.std_mean(loss, unbiased=True)
        self.loss_mean_logs[loss_name] = self.moving_average(
            self.loss_mean_logs[loss_name], mean, self.stats_momentum
        )
        self.loss_sd_logs[loss_name] = self.moving_average(
            self.loss_sd_logs[loss_name], sd, self.stats_momentum
        )

    def flush_logs(self):
        for loss_name, outliers in self.outlier_logs.items():
            fpath = self.dirpath / f"{loss_name}_outliers.tsv"
            with open(fpath, "w", encoding="utf8", newline="") as output_file:
                rows = list(outliers.values())
                rows = sorted(rows, key=lambda row: row["outlier_score"], reverse=True)
                fc = csv.DictWriter(
                    output_file,
                    fieldnames=rows[0].keys(),
                    delimiter="\t",
                )
                fc.writeheader()
                fc.writerows(rows)
