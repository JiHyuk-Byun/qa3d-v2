from typing import List, Dict, Tuple

from torch.utils.data import DataLoader
import numpy as np

from .example_manager import ExamplarManager, Examplar
from .dataset import Gobjaverse280k
from .asset import Asset

class DataManager:
    def __init__(self, src_path: str,
                 meta_path: str,
                 criteria: List[str],
                 n_batch: int,
                 n_workers: int,
                 prefetch: int,
                 pin_memory: bool,
                 example: dict=None):
        super().__init__()
        self.src_path = src_path
        self.meta_path = meta_path
        
        self.n_batch = n_batch
        self.n_workers = n_workers
        self.prefetch = prefetch
        self.pin_memory = pin_memory
        
        self.criteria = criteria
        
        self.example_manager = ExamplarManager(**example, src_path=src_path, criteria=criteria)

        self.data_lst = None
        
    def prepare(self, ):
        self.example_manager.prepare()
    
    def load_dataloader(self):

        dataset = Gobjaverse280k(self.src_path, self.data_lst)
        prefetch_factor = (self.n_batch //self.n_workers) * self.prefetch
        if prefetch_factor < 1:
            prefetch_factor = self.prefetch
        
        return DataLoader(dataset,
                          batch_size= self.n_batch,
                          shuffle=True,
                          collate_fn=self._collate_fn,
                          num_workers=self.n_batch,
                          prefetch_factor=prefetch_factor,
                          pin_memory=self.pin_memory)
        
    def register_gids_to_process(self, split_path, processed_gids: List[str]):
        
        split_gids = open(split_path, 'r').read().strip().split('\n') # strip and readlines
        
        self.data_lst = [gid for gid in split_gids if gid not in processed_gids]
    
    def sample_and_pair_examplars(self, batch)->List[Tuple[Asset, Dict[str, np.ndarray]]]:
        
        batch_sample = self.example_manager.batch_sample(len(batch))
        
        pair_lst = []
        
        for asset, samples in zip(batch, batch_sample):
            for sample in samples:
                pair_lst.append((asset,sample))
        
        return pair_lst
    
    def _collate_fn(self, batch):

        return batch