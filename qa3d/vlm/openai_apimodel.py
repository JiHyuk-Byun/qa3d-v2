import os
import base64
import time

from transformers import AutoProcessor, AutoTokenizer
from openai import OpenAI
from tqdm import tqdm

from .basevlm import BaseVLM

class OpenaiApiModel(BaseVLM):
    def __init__(self, model_name: str,
                 temperature: float,
                 max_tokens: int,
                 n_choices: int,
                 api_key: str=None,
                 tensor_parallel_size: int=1)->None:
        super().__init__(model_name=model_name,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         n_choices=n_choices,
                         api_key = api_key)        
        
        self.client = self._initialize(api_key)
        
    def _initialize(self, api_key):
        return OpenAI(api_key=api_key)
        
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

    # for 1 question in iter * batch
    def forward_vlm_chat(self, batch_inputs):
        
        answers = []
    
        # OPENAI API doesn't use batch-inference
        for message in tqdm(batch_inputs):
            try:
                time.sleep(1)
                answer = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    max_tokens=self.max_tokens,
                    n=self.n_choices
                )
            except Exception as e:
                answer = [str(e)]*self.n_choices
            
            answers.append(answer)
            
        return answers        
    
    def post_process(self, gids, criteria, batch_outputset):
        
        answers = []
        gid_previous = ''
        for gid, criterion, output in zip(gids, criteria, batch_outputset):
            if gid != gid_previous:
                print(f'\n gobjaverse_id: {gid}')
            print(f'\ncriteria: {criterion}')
            
            response = []
            if isinstance(output, list): # Errored
                response = answer
            else: # Typical Answers 
                for i in range(self.n_choices):
                    generated_text = output.choices[i].message.content
                    response.append(generated_text)
                    print(f"response: {generated_text!r}")

            answers.append({'answers': response})
            
            gid_previous = gid
            
        return answers
    # def _make_vlm_input(self, inputs): # List of content
    #     contents = []
    #     gids = []
    #     criteria = []
    #     for input_set in inputs:
    #         contents.append(input_set.prompt)
    #         gids.append(input_set.gid)
    #         criteria.append(input_set.criterion)
            
    #     return gids, criteria, contents

