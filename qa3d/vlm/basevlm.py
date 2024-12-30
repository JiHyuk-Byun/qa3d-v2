import os
from os import path as osp
from abc import abstractmethod, ABC

import jsonlines
from typing import List
        
class BaseVLM(ABC):
    def __init__(self, model_name: str,
                 temperature: float,
                 max_tokens: int,
                 n_choices: int,
                 api_key: str=None,)->None:
        super().__init__()
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_choices = n_choices
        
        self.client = None
        self.processor = None
        self.sampling_params = None
        self.api_key = api_key
    
    def run(self, batch_inputset):
        tgt_gids, criteria, batch_vlm_message = self.make_vlm_input(batch_inputset)
        batch_output = self.forward_vlm(tgt_gids, criteria, batch_vlm_message)

        return batch_output

    def make_vlm_input(self, batch_inputset):
        messages = []
        gids = []
        criteria = []
        
        for input_set in batch_inputset:

            messages.append([{"role": "user",
                              "content": input_set.prompt}])
            gids.append(input_set.gid)
            criteria.append(input_set.criterion)
            
        return gids, criteria, messages
        
    @abstractmethod
    def forward_vlm(self, tgt_gids, criteria, batch_inputs):
        pass
    
    # Create open-ai batch file format
    # def create_openai_batch_file(self, out_dir, batch_prompt: List[List[dict]]):
    #     '''
    #     {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
    #     {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
    #     '''
    #     #for i in messages: dict["body"]["messages"]= i
    #     request_lst = []
    #     for prompt_idx, prompt in enumerate(batch_prompt):
    #         openai_format = {
    #             'custom_id': f'request-{prompt_idx}',
    #             'method': 'POST',
    #             'url': 'v1/chat/completions',
    #             'body': {
    #                 'model': self.model_name,
    #                 'messages': [{
    #                     'role': 'user',
    #                     'content': prompt
    #                 }],
    #                 'max_completion_tokens': self.max_tokens,
    #                 'temperature': self.temperature,
    #                 'n_choices': self.n_choices,
                    
    #             }
    #         }
    #         request_lst.append(openai_format)
            
    #     with jsonlines.open(osp.join(out_dir, 'batch_files', 'openai_batch.jsonl'), mode='w') as writer:
    #         writer.write_all(request_lst)
            
    #     return request_lst