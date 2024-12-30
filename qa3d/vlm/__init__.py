from .vlm_manager import VLMLoader

def load_vlm(model_name, **kwargs):
    
    if kwargs['use_openai_api']:
        OpenaiApiModel( model_name=kwargs['model_name'], 
                        api_key=kwargs['api_key'], 
                        temperature=kwargs['temperature'], 
                        max_tokens=kwargs['max_tokens'], 
                        n_choices=kwargs['n_choices)'])
    else:
        LocalInferModel(model_name=kwargs['model_name'], 
                        temperature=kwargs['temperature'], 
                        max_tokens=kwargs['max_tokens'], 
                        n_choices=kwargs['n_choices'])


    print(f"Model: {kwargs['model_name']}\ntemperature: {kwargs['temperature']}, max_tokens: {kwargs['max_tokens']}, n_choices: {kwargs['n_choices']}")
    
    