from typing import List

from ..data import Asset

class InputSet:
    '''
    gid directory 안에 criterion 폴더 6개 만들고, 그 안에 examplar_gid set * sampling 횟수
    '''
    def __init__(self, criterion: str, gid: str, prompt:List[dict], asset_image, examplar_gids: list):
        
        self.criterion = criterion
        self.gid = gid
        self.prompt = prompt
        self.asset_image = asset_image
        self.examplar_gids = examplar_gids
    