import numpy as np
import torch as T
from torch.utils.data import Dataset


class IrisDataset(Dataset):

    def __init__(self, file_path):
        self.data_x = np.loadtxt(file_path,
                                 usecols=range(0, 4), delimiter=",", dtype=np.float32)
        self.data_y = np.loadtxt(file_path,
                                 usecols=[4], delimiter=",", dtype=np.float64)

    def __getitem__(self, index):
        features = T.Tensor(self.data_x[index])
        labels = T.LongTensor(np.array(self.data_y[index]))
        return (features, labels)

    def __len__(self):
        return len(self.data_x)
