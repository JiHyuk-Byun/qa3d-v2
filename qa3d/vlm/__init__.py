from .openai_apimodel import OpenaiApiModel
from .localmodel import LocalInferModel

def load_vlm(**kwargs):

    if kwargs['use_openai_api']:
        return OpenaiApiModel( model_name=kwargs['model_name'], 
                        api_key=kwargs['api_key'], 
                        temperature=kwargs['temperature'], 
                        max_tokens=kwargs['max_tokens'], 
                        n_choices=kwargs['n_choices'])
    else:
        return LocalInferModel(model_name=kwargs['model_name'], 
                        temperature=kwargs['temperature'], 
                        max_tokens=kwargs['max_tokens'], 
                        n_choices=kwargs['n_choices'])

    print(f"Model: {kwargs['model_name']}\ntemperature: {kwargs['temperature']}, max_tokens: {kwargs['max_tokens']}, n_choices: {kwargs['n_choices']}")
   