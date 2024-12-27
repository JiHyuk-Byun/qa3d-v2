import os
import base64

from openai import OpenAI

from .basemodel import BaseModel

# parsing 해서 널어줘야.
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
    def question_and_answer(self, questions):
        '''
        Returns List of Response. Each Response has n_choices answers from equal examplar set.
        '''
        questions = self._make_llm_input(questions)
        answers = []
        
        for question in questions:
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
                response = None
                
            print("response:", response)
            print("error: ", err)
            answers.append({'answers': answers,
                            'error': err})
            
        return answers
    
    def _make_llm_input(self, inputs): # List of content
        contents = []

        for input_set in inputs:
            contents.append(input_set.prompt)
            
        return contents

    # # ToDo1: response로부터 output 빼기
    # for i in range(n_choices):
    #                 answer = response.choices[i].message.content
    # # ToDo2: batch inference 반영