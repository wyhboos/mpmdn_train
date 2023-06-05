import os
import numpy as np
import torch
# from src.lmpn_src.Lidar_info_process import lidar_info_process
from torch.utils.data import Dataset


# global motion planning network dataset
class GMPNDataset(Dataset):
    """
    x_env:input env info
    x_cur_pos:input current position
    x_goal_pos:input goal position
    y:output next position
    """

    def __init__(self, data_file, env_info_length, data_len=None):
        data = np.load(data_file, allow_pickle=True)
        data = np.array(data, dtype=np.float32)
        if data_len is not None:
            data = data[:data_len, :]
        print("Data shape is:", data.shape)
        self.x_env = data[:, :env_info_length]
        self.x_cur_pos = data[:, env_info_length:env_info_length + 2]
        self.x_goal_pos = data[:, env_info_length + 2:env_info_length + 4]
        self.y = data[:, env_info_length + 4:env_info_length + 6]
        self.index = data[:, env_info_length + 6:env_info_length + 7]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x_env[item], self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]



class GMPNDataset_S2D_Pt(Dataset):
    """
    x_env:input env info
    x_cur_pos:input current position
    x_goal_pos:input goal position
    y:output next position
    """

    def __init__(self, data_file, env_info_length, data_len=None, use_env=True):
        self.use_env = use_env
        data = np.load(data_file, allow_pickle=True)
        data = np.array(data, dtype=np.float32)
        if data_len is not None:
            data = data[:data_len, :]
        print("Data shape is:", data.shape)
        self.x_env = data[:, :env_info_length]
        self.x_cur_pos = data[:, env_info_length:env_info_length + 2]
        self.x_goal_pos = data[:, env_info_length + 2:env_info_length + 4]
        self.y = data[:, env_info_length + 4:env_info_length + 6]
        self.index = data[:, env_info_length + 6:env_info_length + 7]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        if self.use_env:
            return self.x_env[item], self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]
        else:
            return self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]

class GMPNDataset_S2D_RB(Dataset):
    """
    x_env:input env info
    x_cur_pos:input current position
    x_goal_pos:input goal position
    y:output next position
    """

    def __init__(self, data_file, env_info_length, data_len=None, use_env=True):
        self.use_env = use_env
        data = np.load(data_file, allow_pickle=True)
        data = np.array(data, dtype=np.float32)
        if data_len is not None:
            data = data[:data_len, :]
        print("Data shape is:", data.shape)
        self.x_env = data[:, :env_info_length]
        self.x_cur_pos = data[:, env_info_length:env_info_length + 3]
        self.x_goal_pos = data[:, env_info_length + 3:env_info_length + 6]
        self.y = data[:, env_info_length + 6:env_info_length + 9]
        self.index = data[:, env_info_length + 9:env_info_length + 10]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        if self.use_env:
            return self.x_env[item], self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]
        else:
            return self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]


class GMPNDataset_S2D_TL(Dataset):
    """
    x_env:input env info
    x_cur_pos:input current position
    x_goal_pos:input goal position
    y:output next position
    """

    def __init__(self, data_file, env_info_length, data_len=None, use_env=True):
        self.use_env = use_env
        data = np.load(data_file, allow_pickle=True)
        data = np.array(data, dtype=np.float32)
        if data_len is not None:
            data = data[:data_len, :]
        print("Data shape is:", data.shape)
        self.x_env = data[:, :env_info_length]
        self.x_cur_pos = data[:, env_info_length:env_info_length + 4]
        self.x_goal_pos = data[:, env_info_length + 4:env_info_length + 8]
        self.y = data[:, env_info_length + 8:env_info_length + 12]
        self.index = data[:, env_info_length + 12:env_info_length + 13]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        if self.use_env:
            return self.x_env[item], self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]
        else:
            return self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]


class GMPNDataset_S2D_ThreeL(Dataset):
    """
    x_env:input env info
    x_cur_pos:input current position
    x_goal_pos:input goal position
    y:output next position
    """

    def __init__(self, data_file, env_info_length, data_len=None, use_env=True):
        self.use_env = use_env
        data = np.load(data_file, allow_pickle=True)
        data = np.array(data, dtype=np.float32)
        if data_len is not None:
            data = data[:data_len, :]
        print("Data shape is:", data.shape)
        self.x_env = data[:, :env_info_length]
        self.x_cur_pos = data[:, env_info_length:env_info_length + 5]
        self.x_goal_pos = data[:, env_info_length + 5:env_info_length + 10]
        self.y = data[:, env_info_length + 10:env_info_length + 15]
        self.index = data[:, env_info_length + 15:env_info_length + 16]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        if self.use_env:
            return self.x_env[item], self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]
        else:
            return self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]

class GMPNDataset_C3D_Point(Dataset):
    """
    x_env:input env info
    x_cur_pos:input current position
    x_goal_pos:input goal position
    y:output next position
    """

    def __init__(self, data_file, env_info_length, data_len=None, use_env=True):
        self.use_env = use_env
        data = np.load(data_file, allow_pickle=True)
        data = np.array(data, dtype=np.float32)
        if data_len is not None:
            data = data[:data_len, :]
        print("Data shape is:", data.shape)
        self.x_env = data[:, :env_info_length]
        self.x_cur_pos = data[:, env_info_length:env_info_length + 3]
        self.x_goal_pos = data[:, env_info_length + 3:env_info_length + 6]
        self.y = data[:, env_info_length + 6:env_info_length + 9]
        self.index = data[:, env_info_length + 9:env_info_length + 10]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        if self.use_env:
            return self.x_env[item], self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]
        else:
            return self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]

class GMPNDataset_Arm(Dataset):
    """
    x_env:input env info
    x_cur_pos:input current position
    x_goal_pos:input goal position
    y:output next position
    """

    def __init__(self, data_file, env_info_length, data_len=None):
        data = np.load(data_file, allow_pickle=True)
        data = np.array(data, dtype=np.float32)
        if data_len is not None:
            data = data[:data_len, :]
        print("Data shape is:", data.shape)
        self.x_env = data[:, :env_info_length]
        self.x_cur_pos = data[:, env_info_length:env_info_length + 7]
        self.x_goal_pos = data[:, env_info_length + 7:env_info_length + 14]
        self.y = data[:, env_info_length + 14:env_info_length + 21]
        self.index = data[:, env_info_length + 21:env_info_length + 22]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x_env[item], self.x_cur_pos[item], self.x_goal_pos[item], self.y[item], self.index[item]


class CloudDataset(Dataset):
    def __init__(self, data_file, data_len=None):
        data = np.load(data_file, allow_pickle=True)
        data = np.array(data, dtype=np.float32)
        if data_len is not None:
            data = data[:data_len, :]
        print("Data shape is:", data.shape)
        self.cloud = data

    def __len__(self):
        return self.cloud.shape[0]

    def __getitem__(self, item):
        return self.cloud[item, :, :]
