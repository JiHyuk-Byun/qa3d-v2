
from typing import Dict, Tuple
from os import path as osp

import numpy as np
from PIL import Image


TARGET_VIEW_IDX = {40: [0, 6, 12, 18], #round_view = 24
                   52: [0, 9, 18, 27]}              #round_view = 36

def load_all_images(asset_dir: str)->Dict[str, Image.Image]:
    # Get alpha
    rgba = Image.open(osp.join(asset_dir, 'rbg.png')).convert('RGBA')
    a = rgba.channel('A')
    
    # load rendered images
    _rgb = rgba.convert('RGB')
    _albedo = Image.open(osp.join(asset_dir, 'albdo.png')).convert('RGB')
    _normal = Image.open(osp.join(asset_dir, 'normal.png')).convert('RGB')
    _mr = Image.open(osp.join(asset_dir, 'mr.png')).convet('RGB')
    
    # color background
    rgb = _color_background(_rgb, a)
    albedo = _color_background(_albedo, a, (80,80,80))
    normal = _color_background(_normal, a)
    metallic, roughness = _color_background(_mr, a, (80,80,80)).split() # Gray
    
    return {"rgb": rgb,
            "albedo": albedo,
            "normal": normal,
            "metallic": metallic,
            "roughness": roughness}

def _color_background(image, alpha, background_color: Tuple[int]=(256,256,256)):
    
    background = Image.new('RGB', image.size, background_color)
    
    return Image.composite(image, background, alpha)
