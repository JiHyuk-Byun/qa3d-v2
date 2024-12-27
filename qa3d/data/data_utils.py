
from typing import Dict, Tuple
from os import path as osp

import numpy as np
from PIL import Image


TARGET_VIEW_IDX = {40: [0, 6, 12, 18], #round_view = 24
                   52: [0, 9, 18, 27]}              #round_view = 36

def load_all_images(asset_dir: str)->Dict[str, Image.Image]:
    # Get alpha
    rgba = Image.open(osp.join(asset_dir, 'rgb.png')).convert('RGBA')
    a = rgba.getchannel('A')
    
    # load rendered images
    _rgb = rgba.convert('RGB')
    _albedo = Image.open(osp.join(asset_dir, 'albedo.png')).convert('RGB')
    _normal = Image.open(osp.join(asset_dir, 'normal_map.png')).convert('RGB')
    _metallic = Image.open(osp.join(asset_dir, 'metallic_map.png')).convert('RGB')
    _roughness = Image.open(osp.join(asset_dir, 'roughness_map.png')).convert('RGB')
    
    # color background
    rgb = _color_background(_rgb, a)
    albedo = _color_background(_albedo, a, (80,80,80))
    normal = _color_background(_normal, a)
    metallic = _color_background(_metallic, a, (80,80,80))
    roughness = _color_background(_roughness, a, (80,80,80))
    
    
    return {"rgb": rgb,
            "albedo": albedo,
            "normal_map": normal,
            "metallic_map": metallic,
            "roughness_map": roughness}

def _color_background(image, alpha, background_color: Tuple[int]=(256,256,256)):
    
    background = Image.new('RGB', image.size, background_color)
    
    return Image.composite(image, background, alpha)
