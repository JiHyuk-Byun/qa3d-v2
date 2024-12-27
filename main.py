import os
from os import path as osp
from typing import List, Tuple, Dict
import time

import numpy as np
from argparse import ArgumentParser
from omegaconf import OmegaConf
from tqdm import tqdm

from qa3d.vlm import VLMLoader
from qa3d.data import DataManager, Asset, Examplar
from qa3d.stat import StatPilot
from qa3d.prompt import PromptBuilder
from utils.save_answers import save_answers

parser = ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='config/main_gpt4o.yaml')

args = parser.parse_args()

def main():
    cfg = OmegaConf.load(args.config)
    save_dir = osp.join(cfg.experiment.out_dir, "answers")
    os.makedirs(save_dir, exist_ok=True)
# ToDo0: Read target data split for current processing gpu
    
    # 5000개 정도의 각 줄별로 gid가적힌 txt 파일의 path, 그 index 안에서 처리된 gid return 
    
    stat_pilot = StatPilot(**cfg.stat)
    split_idx, split_path = stat_pilot.find_unmarked_split()
    stat_pilot.mark_processing()
    
    processed_gids: List[str] = stat_pilot.get_processed_gids(split_idx)
    print(f"Assigned split: {split_idx}\n Number of processed gids: {len(processed_gids)}")
    
# ToDo1: Load VLM
    vlm_loader = VLMLoader(**cfg.model)
    print("Loading VLM...")
    vlm_loader.load_vlm()
    model = vlm_loader.model
    print(vlm_loader)
# ToDo2: load examplar and build Data Manager
    data_manager = DataManager(**cfg.data, criteria=list(cfg.prompt.input_types.keys()))
    data_manager.prepare()
# ToDo3: Text Prompt build
    prompt_builder = PromptBuilder(**cfg.prompt)
# ToDo4: DataLoader
    data_manager.register_gids_to_process(split_path, processed_gids)
    dataloader = data_manager.load_dataloader()
    n_iterations = len(dataloader)
    start_time = time.time()
# ToDo5: batch QA
    for batch_idx, batch in enumerate(dataloader): # batch: batch of Assets
        for asset in batch:
            asset.load_image_data()
        
        #todo1: batch내 각 item 별로 example pairing 하기
        paired_batch: List[Tuple[Asset, Dict[str, np.ndarray]]] = data_manager.sample_and_pair_examplars(batch)
        
        #todo2: prompt에 image 넣기
        batch_inputset = []
        for asset, criterion_examplar in paired_batch:
            input_set = prompt_builder.insert_images_to_prompt(asset, criterion_examplar) # List of InputSet
            batch_inputset.extend(input_set) # List of InputSet, n_sampling * criteria * batch size
        
        #todo4: batch inference 돌리기
        batch_answers = model.question_and_answer(batch_inputset) 
        end_time = time.time()
        
        # for answer in answers:
        #     print(answer)
        #todo5: answer들을 저장하기. 각 batch 별로: n_choices*num_example_sampling
        print(f'[batch {batch_idx}/{n_iterations}] outputs: {len(batch_answers)},'
              f' time: {end_time - start_time:.2f} s')
        
        save_answers(save_dir, batch_inputset, batch_answers)

        stat_pilot.write_processed_gids([asset.gid for asset in batch])

        
    stat_pilot.mark_finished()

#def _save_answers(save_dir, input_set, answer): # gid 별로 폴더 만들어서, 각 gid안에 example image set과 target image, answer_1-1~5-3.txt 저장
if __name__ == '__main__':
    main()
    

# TODO 1. inputset class 만들어서 batch input 다시 만들기 v
# TODO 2. _make_llm_input 함수 정의하기 v
# TODO 3. question_and_answer 뽑기 v 
# TODO 4. _save_answers
