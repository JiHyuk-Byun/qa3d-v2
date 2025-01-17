import os

from typing import Dict
from transformers import AutoProcessor, AutoTokenizer
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
                 pipeline_parallel_size: int=1,
                 gpu_memory_utilization: float=0.8)->None:
        super().__init__(model_name=model_name,
                         temperature=temperature,
                         max_tokens=max_tokens,
                         n_choices=n_choices,
                         api_key = api_key,
                         tensor_parallel_size = tensor_parallel_size,
                         pipeline_parallel_size = pipeline_parallel_size,
                         gpu_memory_utilization = gpu_memory_utilization)        
        
        self.processor, self.sampling_params, self.llm = self._initialize()
        
    def _initialize(self, 
                    limit_mm_per_prompt: dict={'image': 24}):
        
        return (AutoProcessor.from_pretrained(self.model_name, use_fast=True),
                SamplingParams(n=self.n_choices, temperature=self.temperature, max_tokens=self.max_tokens),
                LLM(model=self.model_name, tensor_parallel_size=self.tensor_parallel_size, pipeline_parallel_size=self.pipeline_parallel_size,
                    gpu_memory_utilization=self.gpu_mem_util, limit_mm_per_prompt=limit_mm_per_prompt))

    def make_vlm_input(self, batch_inputset):
        inputs = []
        gids = []
        criteria = []
        for input_set in batch_inputset:
            criterion, content, image_inputs = input_set.criterion, input_set.prompt, input_set.input_images
            
            messages = [{'role': 'user', 'content': content}]
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            mm_data = {'image': image_inputs}

            inputs.append({
            'prompt': prompt,
            'multi_modal_data': mm_data
        })
            
            gids.append(input_set.gid)
            criteria.append(criterion)
        return gids, criteria, inputs
        
    def forward_vlm_chat(self, batch_inputs):
    
        outputs = self.llm.generate(prompts=batch_inputs, sampling_params=self.sampling_params, use_tqdm=True)
        
        return outputs

    def post_process(self, gids, criteria, batch_outputset):
        
        answers = []
        gid_previous = ''
        for gid, criterion, output in zip(gids, criteria, batch_outputset):
            if gid != gid_previous:
                print(f'\n gobjaverse_id: {gid}')
            print(f'\ncriteria: {criterion}')
            print(f'Length of input token: {len(output.prompt_token_ids)}')
            response = []
            if self.n_choices > 1:
                for i in range(self.n_choices):
                    generated_text = output[i].outputs[0].text
                    response.append(generated_text)
                    print(f"response: {generated_text!r}")
            else:
                generated_text = output.outputs[0].text
                print(f"response: {generated_text!r}")
                
            answers.append({'answers': response})
            
            gid_previous = gid
            
        return answers

    # def forward_vlm_chat(self, tgt_gids, criteria, batch_inputs):
       
    #     outputs = self.llm.chat(messages=batch_inputs,
    #                             sampling_params=self.sampling_params,
    #                             use_tqdm=True)
        
    #     return tgt_gids, criteria, outputs        


    
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