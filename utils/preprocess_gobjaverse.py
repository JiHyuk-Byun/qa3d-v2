import os
from os import path as osp
import json

from PIL import Image
import cv2
import numpy as np
import glob
from argparse import ArgumentParser

TARGET_VIEW_IDX = [0, 6, 12, 18]
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

# TARGET_VIEW_IDX = {40: [0, 6, 12, 18],              #round_view = 24
#                    52: [0, 9, 18, 27]}              #round_view = 36

parser = ArgumentParser()
parser.add_argument('--src_dir', '-d', type=str, default='datasets/gobjaverse_280k_test')
#parser.add_argument('--sample_view', '-v', type=int, default=4)
parser.add_argument('--gids_lst', '-g', type=str, default='datasets/gobjaverse_280k_test.json')#'assets/example.json')

args = parser.parse_args()

def _color_background(image, alpha, background_color =(256,256,256)):
    
    background = Image.new('RGB', image.size, background_color)
    
    return Image.composite(image, background, alpha)

def extract_normal_map(nd_img):
    '''
    Extract normal map from normal depth map.
    '''
    assert len(nd_img.shape) == 3 and nd_img.shape[2] == 4
    # RGBA 분리
    rgb = nd_img[:, :, :3]   # R, G, B 채널
    alpha = nd_img[:, :, 3]  # Alpha 채널

    # 배경 검정색 처리
    # Alpha 값이 0인 부분은 RGB를 (0, 0, 0)으로 설정
    rgb[alpha == 0] = [255, 255, 255]

    return rgb

def create_square_images(img_lst, ncols):
    img_array = []
    img_row_array = []
    for curr_img in img_lst:

        if len(img_row_array) < ncols - 1:
            img_row_array.append(curr_img)
        else: # len(img_row_array) == ncols
            assert len(img_row_array) == ncols - 1

            img_row_array.append(curr_img)
            img_array.append(np.concatenate(img_row_array, axis=1))
            img_row_array = []

    return np.concatenate(img_array, axis=0)
            
def main():
    
    src_dir = args.src_dir
    gids_lst = sorted(json.load(open(args.gids_lst, 'r')), key=lambda x: int(x.split('/')[-2] + x.split('/')[-1]))
    #print(gids_lst)
    print("Number of Assets: ", len(gids_lst))
    for gid in gids_lst:
        
        ## Sample images
        gid_path = osp.join(src_dir, gid)
        all_view = glob.glob(osp.join(gid_path, "*/"))
        n_view = len(all_view)
        all_view = sorted(all_view, key = lambda x: int(x.split('/')[-2]))
        
        sampled_view = [all_view[i] for i in TARGET_VIEW_IDX]

        rgb_lst = []
        albedo_lst = []
        normal_lst = []
        mr_lst = []
        
        for sample in sampled_view:

            name = sample.split('/')[-2]
 

            _rgb = cv2.imread(osp.join(sample, name) + '.png', cv2.IMREAD_UNCHANGED)
            _albedo = cv2.imread(osp.join(sample, name) + '_albedo.png', cv2.IMREAD_UNCHANGED)
            _normal = cv2.imread(osp.join(sample, name) + '_nd.exr', cv2.IMREAD_UNCHANGED) * 255
            _normal = extract_normal_map(_normal)
            _mr = cv2.imread(osp.join(sample, name) + '_mr.png', cv2.IMREAD_UNCHANGED)
            
            
            rgb_lst.append(_rgb)
            albedo_lst.append(_albedo)
            normal_lst.append(_normal)
            mr_lst.append(_mr)
            
        cv2.imwrite(osp.join(gid_path, 'rgb.png'), create_square_images(rgb_lst, 2))
        cv2.imwrite(osp.join(gid_path, 'albedo.png'), create_square_images(albedo_lst, 2))
        cv2.imwrite(osp.join(gid_path, 'normal_map.png'), create_square_images(normal_lst, 2))
        cv2.imwrite(osp.join(gid_path, 'mr.png'), create_square_images(mr_lst, 2))

        
        ## Color bg and split mr
        rgba = Image.open(osp.join(gid_path, 'rgb.png')).convert('RGBA')
        a = rgba.getchannel('A')
        
        # load rendered images
        _rgb = rgba.convert('RGB')
        _albedo = Image.open(osp.join(gid_path, 'albedo.png')).convert('RGB')
        _normal = Image.open(osp.join(gid_path, 'normal_map.png')).convert('RGB')
        _mr = Image.open(osp.join(gid_path, 'mr.png')).convert('RGB')
        
        # color background
        rgb = _color_background(_rgb, a)
        albedo = _color_background(_albedo, a, (80,80,80))
        normal = _color_background(_normal, a)
        mr = _color_background(_mr, a, (80,80,80))
        metallic, roughness, _ = mr.split()
        
        rgb.save(osp.join(gid_path, 'rgb.png'))
        albedo.save(osp.join(gid_path, 'albedo.png'))
        normal.save(osp.join(gid_path, 'normal_map.png'))
        metallic.save(osp.join(gid_path, 'metallic_map.png'))
        roughness.save(osp.join(gid_path, 'roughness_map.png'))
        
    print('Complete!!')

if __name__ == '__main__':
    main()
    