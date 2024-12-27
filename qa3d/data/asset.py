from typing import Dict
from os import path as osp

from PIL import Image

from .data_utils import load_all_images

class Asset:
    def __init__(self, gid: str, src_path):
        super().__init__()
        self.gid: str = gid
        self.path = osp.join(src_path, gid)
        self.image: Dict[str, Image.Image] = None
    
    def __repr__(self):
        return f"gid: {self.gid}, path: {self.path}"
    
    def load_image_data(self):
        self.image =  load_all_images(self.path)