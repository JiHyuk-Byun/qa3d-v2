import os
from os import path as osp
import json

def save_answers(save_dir, batch_inputset, batch_answers):
    '''
    Assume batch_inputset and batch_answers are having same order.
    '''
    for input_set, answer in zip(batch_inputset, batch_answers):
        asset_path = osp.join(save_dir, input_set.gid)
        osp.makedirs(asset_path, exist_ok=True)
        input_info = {}
        output_info = {}
        
        input_info['asset_gid'] = input_set.gid
        input_info[input_set.criterion] ={}
    
        # lower level, higher quality
        for lvl, examplar in enumerate(input_set.examplar_gids):
            sample_idx = 0
            key = f'{sample_idx}-level-{lvl}'  

            # 만약 base_key가 이미 존재한다면 뒤에 숫자를 붙여가며 새로운 키를 찾는다.
            
            while key in input_info[input_set.criterion]:
                sample_idx += 1
                key = f'{sample_idx}-level-{lvl}'  

            # 최종 확정된 키에 examplar 대입
            input_info[input_set.criterion][key] = examplar
            
        with open(osp.join(asset_path, 'input.json'), 'w') as f:
            json.dump(f, input_info, indent=4)

        
        output_info['asset_gid'] = input_set.gid
        output_info[input_set.criterion] = {}
        
        for idx, choice in enumerate(answer):
            sample_idx = 0
            key = f'{sample_idx}-answer-{idx}'
            
            while key in input_info[input_set.criterion]:
                sample_idx += 1
                key = f'{sample_idx}-answer-{idx}'

            output_info[input_set.criterion][key] = choice
        
        with open(osp.join(asset_path, 'output.json'), 'w') as f:
            json.dump(f, output_info, indent=4)
            
        input_set.input_images.save(osp.join(asset_path, 'target_img.png'))
        
        
        