
from typing import List

from ..data import Asset

class InputSet:
    '''
    gid directory 안에 criterion 폴더 6개 만들고, 그 안에 examplar_gid set * sampling 횟수
    '''
    def __init__(self, criterion: str, gid: str, prompt:List[dict], input_images, examplar_gids: list):
        
        self.criterion = criterion
        self.gid = gid
        self.prompt = prompt
        self.input_images = input_images
        self.examplar_gids = examplar_gids
        
        
