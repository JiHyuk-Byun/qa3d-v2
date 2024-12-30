import os
import base64

from openai import OpenAI

from .basevlm import BaseVLM

class OpenaiApiModel(BaseVLM):
    def __init__(self, model_name: str,
                 temperature: float,
                 max_tokens: int,
                 n_choices: int,
                 api_key: str=None,)->None:
        super().__init__(model_name=model_name,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         n_choices=n_choices,
                         api_key = api_key)        
        
        self.client = self._initilize(api_key)
        
    def _initialize(self, api_key):

        client = OpenAI(api_key=api_key)
        
        return client
    
    # for 1 question in iter * batch
    def forward_vlm(self, tgt_gids, criteria, batch_inputs):

        answers = []
        gid_previous = ''
        
        for gid, criterion, message in zip(tgt_gids, criteria, batch_inputs):
            if gid != gid_previous:
                print(f"\ngobjaverse_id: {gid}")
            print(f"\ncriteria: {criterion}")
            
            try:
                answer = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    max_tokens=self.max_tokens,
                    n=self.n_choices
                )

                response = []
                for i in range(self.n_choices):
                    response.append(answer.choices[i].message.content)
            
            except Exception as e:
                response = [str(e)]*self.n_choices
                
            print("response:", response)
            
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

