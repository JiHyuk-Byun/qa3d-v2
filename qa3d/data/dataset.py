from os import path as osp
from typing import List

from torch.utils.data import Dataset

from .asset import Asset

class Gobjaverse280k(Dataset):
    def __init__(self, src_path, data_lst: List[str]):
        super().__init__()
        
        self.src_path = src_path # src path of dataset
        self.items = data_lst # list of gid
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        
        item = self.items[idx]

        return Asset(gid=item, src_path=self.src_path)
        