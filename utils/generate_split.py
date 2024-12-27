import os
from os import path as osp
import random
import json

from argparse import ArgumentParser
import glob


parser = ArgumentParser()
parser.add_argument('--number_per_split', '-n', type=int, default=5000)
parser.add_argument('--src_dir', '-d', type=str, default='datasets/gobjaverse_280k')
parser.add_argument('--gid_lst', '-g', type=str, default=None)#'datasets/gobjaverse_280k.json')

args = parser.parse_args()

def main():
    number_per_split = args.number_per_split
    
    src_dir = args.src_dir
    gid_lst = args.gid_lst
    
    if gid_lst == None:
        batch_lst = os.listdir(src_dir)
        gid_lst = []
        for batch in batch_lst:
            gid_lst.extend(list(map(lambda x: osp.join(batch, x), os.listdir(osp.join(src_dir, batch)))))
    
        with open(src_dir + '.json', 'w') as f:
            json.dump(gid_lst, f, indent=4)
        
    random.shuffle(gid_lst)
    
    len_gids = len(gid_lst)
    start_idx = 0
    idx = 0
    while start_idx < len_gids:    
        end_idx = min(start_idx + number_per_split, len_gids)
        split = gid_lst[start_idx:end_idx]

        with open(f'datasets/splits_test/{idx}.txt', "w") as f:
            f.write('\n'.join(split))
            
        idx += 1
        start_idx = end_idx
        
    print('complete!!')
    
    
if __name__ == '__main__':
    main()