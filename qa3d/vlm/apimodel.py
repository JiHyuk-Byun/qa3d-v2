import os

from openai import OpenAI

from .basemodel import BaseModel, Response

# parsing 해서 널어줘야.
class OpenaiApiModel(BaseModel):
    def __init__(self, model_name:    str,
                 temperature:   float,
                 max_tokens: int,
                 n_choices: int,
                 api_key: str=None,)->None:
        
        self.model_name = model_name
        self.client = self._init_client(api_key)
        self.max_tokens=max_tokens
        
    def _init_client(self, api_key):
        
        client = OpenAI(api_key=api_key)
        
        return client
    
    # for 1 question in iter * batch
    def question_and_answer(self, question, n_choices):
        
        try:
            answer = self.client.chat.completions.create(
                model=self.model_name,
                messages=question,
                max_tokens=self.max_tokens,
                n=n_choices
            )
            
            response = []
            for i in range(n_choices):
                response.append(answer.choices[i].message.content)
                
            err = None
        
        except Exception as e:
            err = e
            response = None
        
        
        return Response(response, err)
    
    # # ToDo1: response로부터 output 빼기
    # for i in range(n_choices):
    #                 answer = response.choices[i].message.content
    # ToDo2: batch inference 반영