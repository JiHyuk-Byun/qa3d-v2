import os
from os import path as osp

import glob
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--tgt_path', '-t', type=str, default='/mnt/volume4/users/join/gobjaverse_chunked/chunk_1')
args = parser.parse_args()

SRC_PATH = '/mnt/volume4/users/datasets/gobjaverse_280k_tar'
TGT_PATH = args.tgt_path
def main():
    print(f"{TGT_PATH} is processing...")
    current_tars = glob.glob(osp.join(SRC_PATH, '*', '*.tar'))
    current_tars = sorted(current_tars, key = lambda x: int(x.split('/')[-2] + x.split('/')[-1].split('.')[-2]))
    
    tgt_tars = glob.glob(osp.join(TGT_PATH, '*', '*.tar'))
    tgt_tars = sorted(tgt_tars, key = lambda x: int(x.split('/')[-2] + x.split('/')[-1].split('.')[-2]))
    cnt = 0
    for tar_path in tqdm(tgt_tars):
        gid_tar = osp.join(tar_path.split('/')[-2], tar_path.split('/')[-1])

        tgt_path = osp.join(SRC_PATH, gid_tar)

        if not osp.exists(tgt_path):
            dirname = osp.dirname(tgt_path)
            os.makedirs(dirname, exist_ok=True)
            
            cmd = f'mv {tar_path} {dirname}'
            os.system(cmd)
            cnt += 1
    print(f'{cnt} objects are moved!')
    
if __name__ == '__main__':
    main()