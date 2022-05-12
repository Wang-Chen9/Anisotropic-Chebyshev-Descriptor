import os.path as osp
from sklearn.model_selection import train_test_split

import torch
# from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_ply
import numpy as np
import scipy.io as sio


class FAUST(InMemoryDataset):
    r"""
    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        super(FAUST, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def process(self):

        print("loading data")
        desc_path = osp.join(self.root, 'ACD' 'tr_reg_{:03d}.mat')
        evecs_path = osp.join(self.root, 'evecs', 'tr_reg_{:03d}.mat')
        labels_path = osp.join(self.root, 'labels', 'tr_reg_{:03d}.mat')

        data_list = []
        for i in range(100):
            x = sio.loadmat(desc_path.format(i))['desc'].astype(np.float32)
            y = sio.loadmat(labels_path.format(i))['labels'].astype(np.int64).flatten() - 1

            shape = sio.loadmat(evecs_path.format(i))['shape'][0,0]
            V = shape['evecs'].astype(np.float32)
            D = shape['evals'].astype(np.float32)
            A = shape['A'].astype(np.float32)

            data = Data(x=torch.from_numpy(x), y=torch.from_numpy(y),
                        D=torch.from_numpy(D),
                        V=torch.from_numpy(V),
                        A=torch.from_numpy(A),
                        # index=int(i),
                        )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data.index = int(i)
            data.name = int(desc_path.format(i).split('/')[-1].split('.')[0].split('_')[-1])
            data_list.append(data)
            # print(i)

        torch.save(self.collate(data_list[0:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:100]), self.processed_paths[1])
