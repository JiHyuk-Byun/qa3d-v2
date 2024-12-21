import os
from os import path as osp
import glob
from argparse import ArgumentParser

from omegaconf import OmegaConf
from tqdm import tqdm

from qa3d.vlm import VLMLoader
from qa3d.data import ExampleManager, DataManager

parser = ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='config/main_gpt4o.yaml')

args = parser.parse_args()

def main():
    cfg = OmegaConf.load(args.config)

    
# ToDo1: Load VLM
    vlm_loader = VLMLoader(**cfg.model)
    print("Loading VLM...")
    VLMLoader.load_vlm()
    model = vlm_loader.model
    print(vlm_loader)
# ToDo2: example set build
    data_manager = DataManager(**cfg.data, criteria=list(cfg.input_types.keys()))
    data_manager.prepare()
# ToDo3: Text Prompt build

# ToDo4: DataLoader

# ToDo5: batch QA


if __name__ == '__main__':
    main()