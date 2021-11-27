from pytorch_lightning_helpers import wandb_media_log

class LoggerScheduler(pl.Callback):
    def __init__(self, log_figure_step):
        super().__init__()
        self.log_figure_step = log_figure_step

    @torch.cuda.amp.autocast(enabled=False)
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.global_step % self.log_figure_step == 0:
            loggerdict = {}
            for i in range(len(outputs)):
                loggerdict.update(wandb_media_log(outputs[i]))

            trainer.logger.experiment.log(
                loggerdict,
                step=trainer.global_step,
                commit=False,
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.global_step % self.log_figure_step == 0:
            trainer.logger.experiment.log(
                wandb_media_log(outputs),
                step=trainer.global_step,
                commit=False,
            )



