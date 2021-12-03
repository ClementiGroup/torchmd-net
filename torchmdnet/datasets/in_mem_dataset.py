from torch_geometric.data import Dataset, Data
import glob
import numpy as np
import torch


class InMemoryDataset(Dataset):
    r"""Dataset class that stores smaller datasets entirely in
    memory to avoid excess I/O operations.

    Args:
        coordglob (string): Glob path for coordinate files.
        forceglob (string): Glob path for force files.
        embedglob (string): Glob path for embedding index files.
    """

    def __init__(self, coordglob, forceglob, embedglob, stride=1):
        self.coordfiles = sorted(glob.glob(coordglob))
        self.forcefiles = sorted(glob.glob(forceglob))
        self.embedfiles = sorted(glob.glob(embedglob))

        print('Coordinates files: ', len(self.coordfiles))
        print('Forces files: ', len(self.forcefiles))
        print('Embeddings files: ', len(self.embedfiles))

        assert len(self.coordfiles) == len(self.forcefiles) == len(self.embedfiles)
        # make index
        self.index = []
        nfiles = len(self.coordfiles)
        self.coord_list = []
        self.force_list = []
        self.embedding_list = []

        for i in range(nfiles):
            cdata = np.load(self.coordfiles[i])[::stride]
            fdata = np.load(self.forcefiles[i])[::stride]
            edata = np.load(self.embedfiles[i]).astype(np.int)
            assert cdata.shape == fdata.shape
            assert cdata.shape[1] == fdata.shape[1] == len(edata)

            size = cdata.shape[0]
            self.coord_list.append(cdata)
            self.force_list.append(fdata)
            self.embedding_list.append(edata)

            self.index.extend(list(zip([i] * size, range(size))))
        print('Combined dataset size {}'.format(len(self.index)))

    def __getitem__(self, idx):
        molid, index = self.index[idx]

        cdata = self.coord_list[molid]
        fdata = self.force_list[molid]
        edata = self.embedding_list[molid].astype(np.int)

        return Data(
            pos=torch.from_numpy(np.array(cdata[index])),
            y=torch.from_numpy(np.array(fdata[index])),
            z=torch.from_numpy(edata)
        )

    def __len__(self):
        return len(self.index)
