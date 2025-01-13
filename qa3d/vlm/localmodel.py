import os

from typing import Dict
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from .basevlm import BaseVLM

#parsing 할 필요없이 알아서 이미지가 위치 찾아서 감.
class LocalInferModel(BaseVLM):
    def __init__(self, model_name: str,
                 temperature: float,
                 max_tokens: int,
                 n_choices: int,
                 api_key: str=None,
                 tensor_parallel_size: int=1,
                 pipeline_parallel_size: int=1)->None:
        super().__init__(model_name=model_name,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         n_choices=n_choices,
                         api_key = api_key,
                         tensor_parallel_size = tensor_parallel_size,
                         pipeline_parallel_size = pipeline_parallel_size)        
        
        self.processor, self.sampling_params, self.llm = self._initialize()
        
    def _initialize(self, 
                     gpu_mem_util: int=0.9, 
                     #max_model_len: int=2048,
                     limit_mm_per_prompt: dict={'image': 24}):
        
        return (AutoProcessor.from_pretrained(self.model_name),
                SamplingParams(n=self.n_choices, temperature=self.temperature, max_tokens=self.max_tokens),
                LLM(model=self.model_name, tensor_parallel_size=self.tensor_parallel_size, pipeline_parallel_size=self.pipeline_parallel_size,
                    gpu_memory_utilization=gpu_mem_util, limit_mm_per_prompt=limit_mm_per_prompt))
    
    def forward_vlm_chat(self, tgt_gids, criteria, batch_inputs):
        outputs = self.llm.chat(messages=batch_inputs,
                                sampling_params=self.sampling_params,
                                use_tqdm=True)
        
        return tgt_gids, criteria, outputs
    
    def post_process(self, gids, criteria, batch_outputset):
        
        answers = []
        gid_previous = ''
        for gid, criterion, output in zip(gids, criteria, batch_outputset):
            if gid != gid_previous:
                print(f'\n gobjaverse_id: {gid}')
            print(f'\ncriteria: {criteria}')
            
            response = []
            for i in range(self.n_choices):
                generated_text = answer.outputs[i].text
                response.append(answer.outputs[i].text)
            print(f"response: {generated_text!r}")

            answers.append({'answers': response})
            
            gid_previous = gid
            
        return answers
        
    # for 1 question in iter * batch
    # def question_and_answer(self, question, input_images):
    #     inputs = []
    #     inputs.append(self._make_llm_input(self.processor, question, input_images))
        
    #     try:
    #         answer = self.client.generate(inputs, sampling_params=self.sampling_params, use_tqdm=True)
    #         err = None
    #         response = []
    #     except Exception as e:
    #         answer = None
    #         err= e
        
    #     return Response(answer, err)
    
    # def _make_llm_input(self, question):
    #     messages = [{'role': 'user', 'content': content}]
    #     prompt = processor.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    #     mm_data = {'image': image_inputs}

    #     return {
    #         'prompt': prompt,
    #         'multi_modal_data': mm_data
    #     }
    
        # outputs = model.generate(inputs, sampling_params=sampling_params, use_tqdm=True) # batch_size * num_criterion * num_histogram
        # texts = [x.outputs[0].text for x in outputs]