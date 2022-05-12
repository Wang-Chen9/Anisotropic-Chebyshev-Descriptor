# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 09:36:11 2021

@author: Michael
"""

import os.path as osp
import numpy as np
import scipy.io as sio

import torch
from torch_geometric.data import InMemoryDataset,Data


class SCAPE(InMemoryDataset):
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
        super(SCAPE, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']


    def process(self):
        files=['files_train.txt','files_test.txt']
        for i in range(len(files)):
            file=osp.join(self.root,files[i])
            data_list = []
            with open(file, 'r') as f:
                names = [line.rstrip() for line in f]
            
            for idx, name in enumerate(names):
                x=sio.loadmat(osp.join(self.root,'ACD',name))['desc'].astype(np.float32)['shape'][0,0]
                y=sio.loadmat(osp.join(self.root,'labels',name))['labels'].astype(np.int64).flatten()-1
                shape=sio.loadmat(osp.join(self.root,'evecs',name))
                V=shape['evecs'].astype(np.float32)
                D=shape['evals'].astype(np.float32)
                A=shape['A'].astype(np.float32)
            
                data=Data(x=torch.from_numpy(x),y=torch.from_numpy(y),
                          D=torch.from_numpy(D),
                          V=torch.from_numpy(V),
                          A=torch.from_numpy(A))
            
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    data.name=int(idx)
                data.index = int(idx)
                data_list.append(data)
            
            torch.save(self.collate(data_list), self.processed_paths[i])