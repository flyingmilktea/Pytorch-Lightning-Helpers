import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from FastSpeech.utils import rgetattr, rsetattr
from pytorch_lightning.callbacks.base import Callback


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


class DataSaver(Callback):
    def __init__(self, data_dirpath, graph_dirpath, save_modes, log, ids):
        super().__init__()
        self.data_dirpath = Path(data_dirpath)
        self.graph_dirpath = Path(graph_dirpath)
        self.data_dirpath.mkdir(parents=True, exist_ok=True)
        self.graph_dirpath.mkdir(parents=True, exist_ok=True)
        self.save_modes = save_modes
        if log == "True":
            self.log_bool = True
        else:
            self.log_bool = False
        self.ids = ids

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, *args, **kwargs):
        if len(outputs) == 0:
            return
        outputs = outputs
        filenames = batch["audiopath"]
        duration_pred = outputs["duration_pred"]
        for i in range(len(duration_pred)):
            filename = filenames[i]
            pred_id = Path(filename).stem
            pred_duration = duration_pred[i].cpu().numpy()
            np.save(f"{self.data_dirpath}/{pred_id}.npy", pred_duration)
        if "save_data" in self.save_modes:
            text_length = batch["text_lens"]
            mel_length = batch["mel_lens"]
            for i in range(len(duration_pred)):
                filename = filenames[i]
                pred_id = Path(filename).stem
                if self.ids is not None:
                    if pred_id not in self.ids:
                        continue
                trellis = outputs["alpha"][i]
                alignment = outputs["imv"][i]
                trellis_with_path = trellis.clone()
                trellis_with_path[alignment.bool()] = float("nan")
                trellis_with_path = trellis_with_path.cpu().numpy()
                plotted = trellis_with_path[: text_length[i], : mel_length[i]]
                np.save(f"{self.data_dirpath}/{pred_id}_trellis.npy", plotted)
                log_delta_e = (
                    outputs["duration"][i].float()[: int(text_length[i])].cpu().numpy()
                )
                np.save(f"{self.data_dirpath}/{pred_id}_duration-dfa.npy", log_delta_e)
        if "draw_graphs" in self.save_modes:
            text_length = batch["text_lens"]
            mel_length = batch["mel_lens"]
            f = plt.figure()
            f.set_figwidth(20)
            f.set_figheight(16)
            plt.clf()
            for i in range(len(duration_pred)):
                filename = filenames[i]
                pred_id = Path(filename).stem
                if self.ids is not None:
                    if pred_id not in self.ids:
                        continue
                print(pred_id)
                flat_phones = batch["misc"][i]["phoneme_extended"]
                flat_phones = re.sub(
                    r"\([a-zA-Z_]+/(.+?)\)", "\g<1>", flat_phones
                ).split(" ")
                log_delta_e = (
                    outputs["duration"][i].float()[: int(text_length[i])].cpu().numpy()
                )
                dur_pred = (
                    outputs["duration_pred"][i]
                    .float()[: int(text_length[i])]
                    .cpu()
                    .numpy()
                )
                x = np.arange(len(dur_pred))
                try:
                    plt.bar(
                        x,
                        height=log_delta_e,
                        log=self.log_bool,
                        alpha=0.5,
                        color="b",
                        label="targets",
                    )
                    plt.bar(
                        x,
                        height=dur_pred,
                        log=self.log_bool,
                        alpha=0.5,
                        color="r",
                        label="pred",
                    )
                    plt.xticks(
                        x,
                        [t if t != " " else "<space>" for t in flat_phones],
                        rotation=-90,
                    )
                except Exception as e:
                    print(f"Error: {str(e)}")
                plt.legend()
                plt.tight_layout()
                plt.draw()
                plt.savefig(f"{self.graph_dirpath}/{pred_id}.png")
                plt.clf()
                trellis = outputs["alpha"][i]
                alignment = outputs["imv"][i]
                trellis_with_path = trellis.clone()
                trellis_with_path[alignment.bool()] = float("nan")
                trellis_with_path = trellis_with_path.cpu().numpy()
                plt.imshow(
                    trellis_with_path[: text_length[i], : mel_length[i]],
                    origin="lower",
                    aspect="auto",
                )
                plt.colorbar()
                plt.draw()
                plt.savefig(f"{self.graph_dirpath}/{pred_id}_trellis.png")
                plt.clf()
            plt.close()
