import os
import numpy as np
import torch
from torch.utils.data import Dataset


# Lidar dataset for CAE
class CAE_cloud_Dataset(Dataset):
    def __init__(self, data_file, data_clx=None, data_len=None):
        data = np.load(data_file, allow_pickle=True)
        self.data = np.array(data, dtype=np.float32)
        if data_len is not None:
            self.data = self.data[:data_len]
        if data_clx is not None:
            self.data = self.data[:, :data_clx]
        print("Data shape is:", self.data.shape)
        # self.data = self.data
        print(self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
