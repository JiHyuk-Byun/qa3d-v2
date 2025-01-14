import os
from os import path as osp

from argparse import ArgumentParser
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# Qwen2-vl 모델에 맞는 토크나이저 로드
parser = ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='config/main_gpt4o.yaml')

args = parser.parse_args()

def main():
    cfg = OmegaConf.load(args.config)
    
    data_manager = DataManager(**cfg.data, criteria=list(cfg.prompt.input_types.keys()))
    data_manager.prepare()
    
    prompt_builder = PromptBuilder(**.cfg.prompt, show_prompt=True)
    
    tokenizer = AutoTokenizer.from_pretrained("qwen2-vl")

    # 입력 텍스트
    input_text = "What is the capital of France?"

    # 토큰화
    tokens = tokenizer(input_text, return_tensors="pt")  # PyTorch 텐서로 반환
    input_ids = tokens["input_ids"]  # 토큰 ID
    token_length = input_ids.shape[1]  # 토큰 길이 계산

    print(f"Input text: {input_text}")
    print(f"Token length: {token_length}")