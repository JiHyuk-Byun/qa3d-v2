import os
from os import path as osp
import time
from typing import List

import glob
import requests

THRESHOLD = 60*60*3

class StatPilot:
    def __init__(self, src_path: str, out_dir: str):
        super().__init__()
        
        self.src_path = src_path
        self.out_dir = out_dir
        
        self.ip, self.gpu_id, self.pid = self._get_local_info()
        self.start_time = f'{int(time.time())}'
        
        self.split = None
        self.split_path = None
        
        self.marks_dir = osp.join(self.out_dir, 'marks')
        self.splits_dir = osp.join(self.out_dir, 'splits')
        self._init_stat_dirs()

    def _init_stat_dirs(self):   
        os.makedirs(osp.join(self.marks_dir, "processing"), exist_ok=True)
        os.makedirs(osp.join(self.marks_dir, "finished"), exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)
                
    def find_unmarked_split(self):
        split_lst = glob.glob(osp.join(self.src_path, "*.txt"))
        split_lst = sorted(split_lst, key = lambda x: int(osp.basename(x).split('.')[-2]))
        split_indices = list(map(lambda x: osp.basename(x).rstrip('.txt'), split_lst))
        
        for idx, path in zip(split_indices, split_lst):
            ip, gpu_id, pid, time, status = self._check_status(idx)
            
            if status == "finished":
                continue
                
            elif status == "processing":
                stuck_time = int(self.start_time) - int(time)

                if  stuck_time < THRESHOLD: # Processing...
                    continue
                else:
                    print(f'[IP: {ip}, gpu_id: {gpu_id}, pid: {pid}] is stuck for {stuck_time/3600:.2f} hours.')
                    error_path = osp.join(self.marks_dir, "error", f'{idx}.error')
                    with open (error_path, 'w') as f:
                        f.write(f'[IP: {ip}, gpu_id: {gpu_id}, pid: {pid}] is stuck for {stuck_time/3600:.2f} hours.')
                        
            # Now. Found target split.
            self.split = idx
            self.split_path = osp.join(self.splits_dir, f"{idx}.txt")
            
            return (idx, path)
            
    def _get_local_info(self):
        external_ip = requests.get('https://api.ipify.org') # get external ip
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        pid = os.getpid()
        
        return external_ip.text, gpu_id, str(pid)
    
    def _get_processing_info(self, path):
        with open(path, 'r') as f:
           info = f.read().strip().split("\n")
        if len(info) == 0:
            return None, None, None, None
        else:
            ip, gpu_id, pid, time = info
            return ip, gpu_id, pid, time
    
    def _check_status(self, split_idx):

        path_finished = osp.join(self.marks_dir, "finished", f'{split_idx}.finished')
        path_processing = osp.join(self.marks_dir, "processing", f'{split_idx}.processing')
        ip, gpu_id, pid, time = [None, None, None, None]
        status = None
        
        #is finished
        if osp.exists(path_finished):
            status = 'finished'
        #is processing       
        elif osp.exists(path_processing):
            ip, gpu_id, pid, time = self._get_processing_info(path_processing)

            if [ip, gpu_id, pid, time] != [None, None, None, None]:
                status = 'processing'
        
        return ip, gpu_id, pid, time, status
        
    def mark_processing(self):
        path_processing = osp.join(self.marks_dir, "processing", f'{self.split}.processing')
        
        with open(path_processing, 'w') as f:
            info = [self.ip, self.gpu_id, self.pid, self.start_time]
            f.write("\n".join(info))
        
        self.write_processed_gids([""])
            
            
    def mark_finished(self):
        path_processing = osp.join(self.marks_dir, "processing", f'{self.split}.processing')
        path_finished = osp.join(self.marks_dir, "finished", f'{self.split}.finished')
        
        os.system(f'mv {path_processing} {path_finished}')
    
    def write_processed_gids(self, batch: List[str]):
        
        with open(self.split_path, 'w') as f:
            gids = '\n'.join(batch)
            f.write(gids)
            
    def get_processed_gids(self, split_idx):
        
        split_path = osp.join(self.splits_dir, f'{split_idx}.txt')
        with open(split_path, 'r') as f:
            
            processed_gids = f.read()
            if len(processed_gids) == 0:
                return []
            
            else:
                processed_gids = processed_gids.strip().split('\n')
            
            return processed_gids