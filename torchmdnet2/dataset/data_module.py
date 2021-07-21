
import os
import pytorch_lightning as pl
from torch_geometric.data import DataLoader
from argparse import Namespace

from ..utils import make_splits, TestingContext
from .chignolin import ChignolinDataset


dataset_map = {
    'chignolin': ChignolinDataset,
}

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, dataset_root: str,
                 log_dir: str,
                val_ratio: float = 0.1,
                test_ratio: float = 0.1,
                splits: str = None, batch_size: int = 512,
                inference_batch_size: int = 64, num_workers: int = 1,
                train_stride: int = 1) -> None:
        super(DataModule, self).__init__()
        assert dataset_name in dataset_map, f'Name of the dataset should be one of {list(dataset_map.keys())}'
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name

        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.log_dir = log_dir
        self.splits = splits
        self.train_stride = train_stride
        self.splits_fn = os.path.join(self.log_dir, 'splits.npz')

    def prepare_data(self):
        # make sure the dataset is downloaded
        dataset = dataset_map[self.dataset_name](self.dataset_root)
        # setup the train/test/val split
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(dataset),
            self.val_ratio,
            self.test_ratio,
            filename=self.splits_fn,
            splits=self.splits,
        )



    def setup(self, stage = None):
        dataset = dataset_map[self.dataset_name](self.dataset_root)
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(dataset),
            self.val_ratio,
            self.test_ratio,
            splits=self.splits_fn,
        )

        self.train_dataset = dataset[self.idx_train[::self.train_stride]]
        self.val_dataset = dataset[self.idx_val]
        self.test_dataset = dataset[self.idx_test]


    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, 'train')

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, 'val')

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, 'test')

    def _get_dataloader(self, dataset, stage):
        if stage == 'train':
            batch_size = self.batch_size
            shuffle = True
        elif stage in ['val', 'test']:
            batch_size = self.inference_batch_size
            shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )