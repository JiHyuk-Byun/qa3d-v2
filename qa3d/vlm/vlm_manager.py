import os

from typing import List

from .localmodel import LocalInferModel
from .apimodel import OpenaiApiModel

class VLMLoader:
    def __init__(self, model_name:    str, 
                 use_openai_api:    bool, 
                 temperature:   float,
                 max_tokens:    int,
                 n_choices:     int,
                 api_key: str=None,):
        super().__init__()
        
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_choices = n_choices
        
        self.use_openai_api = use_openai_api
        self.api_key = api_key
        
        self.model = None
    
    def __repr__(self):
        return f"Model: {self.model_name}\ntemperature: {self.temperature}, max_tokens: {self.max_tokens}, n_choices: {self.n_choices}"
    
    def load_vlm(self):
        
        self.model = OpenaiApiModel(self.model_name, self.api_key, self.temperature, self.max_tokens, self.n_choices) if self.use_openai_api \
                else LocalInferModel(self.model_name, self.temperature, self.max_tokens, self.n_choices)