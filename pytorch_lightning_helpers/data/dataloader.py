"""Dataset for reconstruction scheme."""


import pytorch_lightning as pl
from loguru import logger
from nonechucks import SafeDataset
from torch.utils.data import DataLoader, Dataset


class StatelessDataset(Dataset):
    """Dataset for stateless loading.
    Counter examples include speaker embedder and mel-scale matrix
    """

    def __init__(self, dataset, _load_datapath, _load_data):
        self.datalist = []
        self._load_data = _load_data
        for i in dataset:
            self.datalist.extend(_load_datapath(**i))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        return self._load_data(*self.datalist[index])


class MultiStageDataModule(pl.LightningDataModule):
    """DataModule with multiple stages for different data
    """

    def __init__(self, traindms, valdm, safe=False):
        super().__init__()
        self.traindms = traindms
        self.valdm = valdm
        if safe:
            self.traindms = list(map(SafeDataset, self.traindms))
            self.valdm = SafeDataset(self.valdm)
        self.current_dm = traindms[0]

    def train_dataloader(self):
        logger.debug(f"current batch_size is: {self.current_dm['batch_size']}")
        return DataLoader(**self.current_dm)

    def val_dataloader(self):
        return DataLoader(**self.valdm)

    def set_phase(self, stageindex):
        logger.debug(f"setting phase: {stageindex}")
        self.current_dm = self.traindms[stageindex]
