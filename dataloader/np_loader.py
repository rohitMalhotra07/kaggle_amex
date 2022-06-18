import torch
import pandas as pd
from torch.utils.data import Dataset


class NumpyDataDataset(Dataset):
    def __init__(self, X,y):
        self.X = X
        self.y =y

    def __getitem__(self, idx):
        x = self.X[idx]
        target = self.y[idx]
        
        return torch.tensor(x).float(), torch.tensor(target).float()

    def __len__(self):
        return len(self.X)
    

    
class NumpyDataDatasetInference(Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        x = self.X[idx]
        
        return torch.tensor(x).float()

    def __len__(self):
        return len(self.X)