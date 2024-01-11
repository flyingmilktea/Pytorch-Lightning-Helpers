"""Dataset for reconstruction scheme."""


import lightning as L


class DataModule(L.LightningDataModule):
    def __init__(
        self, train_dataloader, val_dataloader, train_dataset, val_dataset, safe=False
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataloader_internal = train_dataloader
        self.val_dataloader_internal = val_dataloader

    def train_dataloader(self):
        return self.train_dataset.iter_torch_batches(**self.train_dataloader_internal)

    def val_dataloader(self):
        return self.val_dataset.iter_torch_batches(**self.val_dataloader_internal)
