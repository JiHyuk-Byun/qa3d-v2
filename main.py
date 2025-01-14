import os
from os import path as osp
from typing import List, Tuple, Dict
import time, datetime

import numpy as np
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from qa3d.vlm import load_vlm
from qa3d.data import DataManager, Asset, Examplar
from qa3d.stat import StatPilot
from qa3d.prompt import PromptBuilder
from utils.save_answers import save_answers

now = datetime.datetime.now()

parser = ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='config/main_gpt4o.yaml')
parser.add_argument('--show_prompt', '-p', type=bool, default=False)

args = parser.parse_args() 

def main():
    cfg = OmegaConf.load(args.config)
    formatted_date = now.strftime("%Y-%m-%d")
    save_dir = osp.join(cfg.experiment.out_dir.replace('<DATE>', formatted_date), "answers")
    os.makedirs(save_dir, exist_ok=True)
    
    #1. Read target data split for current processing gpu
    stat_dir = osp.join(cfg.experiment.out_dir.replace('<DATE>', formatted_date), "stat")
    stat_pilot = StatPilot(**cfg.stat, out_dir=stat_dir)
    split_idx, split_path = stat_pilot.find_unmarked_split()
    stat_pilot.mark_processing()
    
    processed_gids: List[str] = stat_pilot.get_processed_gids(split_idx)
    print(f"Assigned split: {split_idx}\nNumber of processed gids: {len(processed_gids)}")
    print(f"Output path: {save_dir}")
    print("="*46)
    
    #2. Load VLM
    print("\nLoading VLM...")
    model = load_vlm(**cfg.model)
    print("="*46)
    
    #3. load examplar and build Data Manager
    print("\nLoading data and prompts...")
    data_manager = DataManager(**cfg.data, criteria=list(cfg.prompt.input_types.keys()))
    data_manager.prepare()
    
    #4. Text Prompt build
    prompt_builder = PromptBuilder(**cfg.prompt, show_prompt=args.show_prompt)
    
    #5. DataLoader
    data_lst = data_manager.register_gids_to_process(split_path, processed_gids)
    print(f"=========Number of gids to process: {len(data_lst)}=========")
    
    dataloader = data_manager.load_dataloader()
    n_iterations = len(dataloader)
    
    #6. batch QA
    for batch_idx, batch in enumerate(dataloader): # batch: batch of Assets
        start_time = time.time()
        for asset in batch:
            asset.load_image_data()
        
        #6-1. Pairs with sampled examples for each items in batch
        paired_batch: List[Tuple[Asset, Dict[str, np.ndarray]]] = data_manager.sample_and_pair_examplars(batch)
        
        #6-2. Insert image into the prompt.
        batch_inputset = []
        for asset, criterion_examplar in paired_batch:
            input_set = prompt_builder.insert_images_to_prompt(asset, criterion_examplar) # List of InputSet

            batch_inputset.extend(input_set) # List of InputSet, n_sampling * criteria * batch size
        
        #6-3. batch inference.
        batch_answers = model.run(batch_inputset) 
        end_time = time.time()
        
        #6-4. Save answers. for each gid, n_choices*num_example_sampling answers
        print(f'[batch {batch_idx+1}/{n_iterations}] outputs: {len(batch_answers)},'
             f' time: {end_time - start_time:.2f} s')
        
        save_answers(save_dir, batch_inputset, batch_answers)

        stat_pilot.write_processed_gids([asset.gid for asset in batch])
        time.sleep(5)
    stat_pilot.mark_finished()

if __name__ == '__main__':
    main()
    

