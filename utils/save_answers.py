import os
from os import path as osp
import json

from PIL import Image

def _read_json_if_exists(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return {}
    
def save_answers(save_dir, batch_inputset, batch_answers):
    '''
    Assume batch_inputset and batch_answers are having same order.
    '''
    for input_set, answer in zip(batch_inputset, batch_answers):
        asset_path = osp.join(save_dir, input_set.gid)
        os.makedirs(asset_path, exist_ok=True)
        
        # -------------------------
        # 1) input_info 불러오기
        # -------------------------
        input_json_path = osp.join(asset_path, 'input.json')
        input_info = _read_json_if_exists(input_json_path)
        
        # 만약 처음 생성되는 경우라면 초기 구조를 잡아준다.
        if 'asset_gid' not in input_info:
            input_info['asset_gid'] = input_set.gid
        
        # input_info에 특정 criterion 키가 없다면 초기화
        if input_set.criterion not in input_info:
            input_info[input_set.criterion] = {}
        
        # examplar 데이터를 추가
        for lvl, examplar in enumerate(input_set.examplar_gids):
            sample_idx = 0
            key = f'{sample_idx}-level-{lvl}'  
            while key in input_info[input_set.criterion]:
                sample_idx += 1
                key = f'{sample_idx}-level-{lvl}'
            input_info[input_set.criterion][key] = examplar
        
        # 변경된 input_info를 다시 파일에 저장
        with open(input_json_path, 'w') as f:
            json.dump(input_info, f, indent=4)

        # -------------------------
        # 2) output_info 불러오기
        # -------------------------
        output_json_path = osp.join(asset_path, 'output.json')
        output_info = _read_json_if_exists(output_json_path)
        
        if 'asset_gid' not in output_info:
            output_info['asset_gid'] = input_set.gid

        if input_set.criterion not in output_info:
            output_info[input_set.criterion] = {}
        
        for idx, choice in enumerate(answer['answers']):
            sample_idx = 0
            key = f'{sample_idx}-answer-{idx}'
            while key in output_info[input_set.criterion]:
                sample_idx += 1
                key = f'{sample_idx}-answer-{idx}'
            output_info[input_set.criterion][key] = choice
        
        with open(output_json_path, 'w') as f:
            json.dump(output_info, f, indent=4)
        
        # -------------------------
        # 3) target image 저장
        # -------------------------
        images = list(input_set.asset_image.values())  # dict의 value들
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.width
                
        new_im.save(osp.join(asset_path, 'target_img.png'))

            
        
        