import os
from os import path as osp

import glob


def aggregate_splits(src_path):
    
    split_idx = 0
    aggregated_gids = []
    
    while True:
        
        split_path = os.path.join(src_path, f'{split_idx}.txt')
        if not osp.exists(split_path):
            break
        
        with open(split_path, 'r') as f:
            gids_lst = f.read().strip().split('\n')
        
        aggregated_gids.extend(gids_lst)
        
        split_idx +=1
        
    
    with open(osp.join(src_path, 'all.txt'), 'w') as f:
        
        f.write('\n'.join(aggregated_gids))

if __name__ == '__main__':
    src_path = 'datasets/splits_test2'
    aggregate_splits(src_path)