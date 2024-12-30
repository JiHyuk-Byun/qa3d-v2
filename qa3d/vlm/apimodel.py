import os
import base64

from openai import OpenAI

from .basemodel import BaseModel

class OpenaiApiModel(BaseModel):
    def __init__(self, model_name:    str,
                 temperature:   float,
                 max_tokens: int,
                 n_choices: int,
                 api_key: str=None,)->None:
        super().__init__(model_name=model_name,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         n_choices=n_choices,
                         api_key = api_key)        
        self.client = self._init_client(api_key)
        
    def _init_client(self, api_key):

        client = OpenAI(api_key=api_key)
        
        return client
    
    # for 1 question in iter * batch    
    def forward_vlm(self, batch_inputs):

        tgt_gids, criteria, questions = self.make_vlm_input(questions)
        answers = []
        gid_previous = ''
        
        for gid, criterion, question in zip(tgt_gids, criteria, questions):
            if gid != gid_previous:
                print(f"\ngobjaverse_id: {gid}")
            print(f"\ncriteria: {criterion}")
            
            conversation_history = [{
                "role": "user",
                "content": question,
            }]
            try:
                answer = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation_history,
                    max_tokens=self.max_tokens,
                    n=self.n_choices
                )

                response = []
                for i in range(self.n_choices):
                    response.append(answer.choices[i].message.content)
                    
                err = None
            
            except Exception as e:
                err = e
                response = [str(e)]*self.n_choices
                
            print("response:", response)
            
            answers.append({'answers': response,
                            'error': err})
            
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

