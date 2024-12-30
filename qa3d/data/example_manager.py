import os
from os import path as osp
import json
import random

import numpy as np
from typing import List, Dict

from .asset import Asset

class Examplar(Asset):
    def __init__(self, gid: str, src_path: str, score: Dict[str, float]):
        super().__init__(gid, src_path)
        
        self.score = score

class ExamplarManager:
    def __init__(self,
                 num_level: int,
                 num_example_sampling: int,
                 sample_interval: int,
                 sample_offset: int,
                 meta_path: str,
                 src_path: str,
                 criteria: List[str])->None:
        super().__init__()        

        self.num_level = num_level
        self.num_example_sampling = num_example_sampling
        self.sample_interval = sample_interval
        self.sample_offset = sample_offset
        self.src_path = src_path
        self.criteria = criteria
        
        self.examplars:List[Examplar] = self._load_examples(meta_path)

    def _load_examples(self, meta_path)->List[Examplar]:
        examplars = json.load(open(meta_path, "r"))
        
        return [Examplar(examplar['metadata']['gobjaverse_index'], self.src_path, examplar['score']) for examplar in examplars]
    
    # In descending order
    def _sort_by_criterion(self, criterion: str)->List[Examplar]:
        sorted_examplars = sorted(self.examplars, key=lambda x: -x.score[criterion])
        
        return sorted_examplars
    
    def prepare(self):
        for examplar in self.examplars:
            examplar.load_image_data()
        
    # def sample_example_set(self, criterion)->List[Asset]:
    #     sorted_examplars = self._sort_by_criteria(criterion)
        
    #     examplars_size = len(self.examplars)
    #     size_per_level = examplars_size // self.num_level
    #     sample_indices = np.zeros((self.num_level, self.num_example_sampling), dtype=int)
    #     for i in range(self.num_level):
    #         histogram_size = size_per_level- self.sample_interval
    #         level_indices = np.random.choice(histogram_size, self.num_example_sampling, replace=False)
    #         level_indices += i * size_per_level + self.sample_offset
    #         sample_indices[i] = level_indices
    #     sample_indices = np.clip(sample_indices, 0, examplars_size - 1)
        
    #     examplars_lst = []
    #     for level_indices in sample_indices.T:
    #         examplars_lst = examplars_lst.append(sorted_examplars[level_indices])
            
    #     return examplars_lst
    
    def batch_sample(self, n_batch)->List[List[Dict[str, np.ndarray]]]:#batch->n_sampling->criteria->n_level(ndarray of Examplar)
        n_criteria = len(self.criteria)
        examplars_size = len(self.examplars)
        size_per_level = examplars_size // self.num_level
        histogram_size = size_per_level - self.sample_interval

        batch_indices = np.zeros((self.num_level, self.num_example_sampling, n_criteria, n_batch))
        
        # Batch Random Sampling for [n_sampling, n_criteria, n_batch] within range 0~histogram_size
        for i in range(self.num_level):
            level_indices = np.random.randint(low=0, high=histogram_size, size=(self.num_example_sampling, n_criteria, n_batch))
            level_indices += i * size_per_level + self.sample_offset
            
            batch_indices[i] = level_indices
        
        batch_indices = np.clip(batch_indices, 0, examplars_size - 1) # [n_level, n_sampling, n_criteria, n_batch]
        
        batch_indices = batch_indices.transpose(2,0,1,3).astype(int) # [n_criteria, n_level, n_sampling, n_batch]
        
        # Advanced Indexing
        sample = {}
        for criterion, criterion_indices in zip(self.criteria, batch_indices):

            sorted_examplars = np.array(self._sort_by_criterion(criterion))
            criterion_samples = sorted_examplars[criterion_indices] # [n_level, n_sampling, n_batch]
            sample[criterion] = criterion_samples
        
        batch_sample = []
        # Reformatting to batch->n_sampling->criteria->n_level
        for batch_idx in range(n_batch):
            examplar_lst = []
            for sampling_idx in range(self.num_example_sampling):
                sampling_dict = {}
                for criterion in self.criteria:
                    sampling_dict[criterion] = sample[criterion][:, sampling_idx, batch_idx]
                examplar_lst.append(sampling_dict)
            
            batch_sample.append(examplar_lst)
            
        return batch_sample
            