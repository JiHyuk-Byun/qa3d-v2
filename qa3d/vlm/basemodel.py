import os

from typing import List


class Response:
    def __init__(self, response: List[str], err: str):
        super().__init__()
        
        self.response = response
        self.error = err

class BaseModel:
    
    def __init__(self, model_name:    str,
                 temperature:   float,
                 max_tokens: int,
                 api_key: str=None,)->None:
        super().__init__()
        
        self.model_name = None
        self.client = None
        self.max_tokens = None
    
    def question_and_answer(model_name, question: List[str], input_images, n_choices) -> Response:
        pass
        

    # Create open-ai batch file format
    def _create_batch_file(messages: List[str]):
        '''
        {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
        {"custom_id": "request-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "system", "content": "You are an unhelpful assistant."},{"role": "user", "content": "Hello world!"}],"max_completion_tokens": 1000}}
        '''
        #for i in messages: dict["body"]["messages"]= i
        pass