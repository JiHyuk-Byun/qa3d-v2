import os

from typing import Dict
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from .basemodel import BaseModel, Response

#parsing 할 필요없이 알아서 이미지가 위치 찾아서 감.
class LocalInferModel(BaseModel):
    
    def __init__(self, model_name:    str,
                 temperature:   float,
                 max_tokens: int,
                 n_choices: int,
                 api_key: str=None,
                 )->None:
        
        
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = LLM(model=model_name, gpu_memory_utilization=0.8, pipeline_parallel_size=1,
                    max_model_len=1,limit_mm_per_prompt={'image': 1},)#WHY?

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        
    def _make_llm_input(processor, content, input_images):
        messages = [{'role': 'user', 'content': content}]
        
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        mm_data = {'image': input_images}

        return {
            'prompt': prompt,
            'multi_modal_data': mm_data
        }
        
    
    # for 1 question in iter * batch
    def question_and_answer(self, question, input_images, n_choices):
        inputs = []
        inputs.append(self._make_llm_input(self.processor, question, input_images))
        
        try:
            answer = self.client.generate(inputs, sampling_params=self.sampling_params, use_tqdm=True)
            err = None
            response = []
        except Exception as e:
            answer = None
            err= e
        
        return Response(answer, err)
    
    
        # outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=True) # batch_size * num_criterion * num_histogram
        # texts = [x.outputs[0].text for x in outputs]