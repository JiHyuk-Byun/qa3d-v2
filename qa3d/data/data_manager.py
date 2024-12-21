
from typing import List, Dict

from .example_manager import ExamplarManager
from .asset import Asset

class DataManager:
    def __init__(self, src_path: str,
                 meta_path: str,
                 criteria: List[str],
                 example: dict=None):
        
        self.src_path = src_path
        self.meta_path = meta_path
        
        self.example_manager = ExamplarManager(**example, src_path=src_path, criteria=criteria)

        
    def prepare(self, criteria):
        self.example_manager.prepare()
            