import os
from os import path as osp
import json

from typing import List, Dict

from .asset import Asset

class Examplar(Asset):
    def __init__(self, gid: str, src_path: str, score: Dict[str, float]):
        super().__init__(gid, src_path)
        
        self.score = score

class ExamplarManager:
    def __init__(self, 
                 num_example: int,
                 iter_per_example: int,
                 meta_path: str,
                 src_path: str,
                 criteria: List[str])->None:
        
        self.examplars = self._load_examples(meta_path)
        self.num_example = num_example
        self.iter_per_example = iter_per_example
        self.src_path = src_path
        self.criteria = criteria
        
    def _load_examples(self, meta_path)->List[Examplar]:
        examplars = json.load(open(meta_path, "r"))
        
        return [Examplar(examplar['metadata']['gobjaverse_index'], self.src_path, examplar['score']) for examplar in examplars]
    
    def sort_by_criteria(self, criterion: str)->List[]:
        
        sorted_examplars = sorted(self.examplars, key=lambda x: -x.score[criterion])
        
        return sorted_examplars
    
    def prepare(self):
        for examplar in self.examplars:
            examplar.load_image_data()
        
    def create_example_set(self, crierion)->List[Asset]:
        pass