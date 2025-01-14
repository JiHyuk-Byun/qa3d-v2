from os import path as osp
from typing import Literal, Dict, List
import base64

import numpy as np

from ..data import Examplar, ExamplarManager
from ..data import Asset
from .input_set import InputSet

class PromptBuilder:
    def __init__(self, template_dir: str,
                 prompt_type: Literal['scoring', 'indexing'],
                 input_types: Dict[str, List[str]],
                 show_prompt: bool,
                 use_example: bool):
        super().__init__()
        
        self.use_example = use_example
        self.prompt_type = prompt_type if use_example else 'no_examples'
        self.input_types = input_types 

        self.template_dir = template_dir
        self.criteria = list(input_types.keys())
        self.main_structure, self.criteria_instructions = self._read_templates()
                                
        self.show_prompt = show_prompt        
        self.prompt: dict = self._generate_text_prompt()

    def _read_templates(self):
        with open(osp.join(self.template_dir, self.prompt_type, 'main_structure.txt')) as f:
            main_structure = f.read().strip()
            
        with open(osp.join(self.template_dir, self.prompt_type, 'criteria_instructions.txt')) as f:
            criteria_instructions = f.read().strip().split('\n')
        
        return main_structure, criteria_instructions
    
    def _generate_text_prompt(self):
        prompts = {}
        
        for (criterion, input_type), instruction in zip(self.input_types.items(), self.criteria_instructions):
            input_text = self.main_structure
            # insert input types
            input_text = input_text.replace('<|input_types|>', ' and '.join(input_type))
            # insert instructions
            input_text = input_text.replace('<|instruction|>', instruction)
            
            if self.show_prompt:
                print(f'Criterion: {criterion}\nPrompt:\n{input_text}')
            
            text_prompt = [{'type': 'text',
                            'text': input_text}]
            prompts[criterion] = text_prompt
        
        return prompts
    
    # Todo insert example sets
    def insert_images_to_prompt(self, 
                                asset: Asset, 
                                criterion_examplar: Dict[str, np.ndarray])-> list:
        '''
        returns List of InputSet
        '''
        inputs = []
    
        for criterion, examplars in criterion_examplar.items():
            input_images = []
            examplar_gids = []
            criterion_prompt = self.prompt[criterion].copy() # must be not modified
            
            # Insert examplar
            # lower level, higher score.
            if self.use_example:
                for level, examplar in enumerate(examplars):
                    examplar_gids.append(examplar.gid)
                    examplar_caption = self._get_examplar_caption(level)
                    examplar_img_prompt, examplar_img = self._get_image_prompt(examplar, criterion)

                    criterion_prompt.extend([*examplar_caption, *examplar_img_prompt])
                    input_images.extend(examplar_img)
                
            # Insert target asset
            asset_caption = [{'type': 'text',
                              'text': '# Target object:\n'}]
            asset_img_prompt, asset_img = self._get_image_prompt(asset=asset, criterion=criterion)
            
            criterion_prompt.extend([*asset_caption, *asset_img_prompt])
            input_images.extend(asset_img)

            input_set = InputSet(criterion, asset.gid, criterion_prompt, asset.image, input_images, examplar_gids)
            #input_set.print_prompt()
            inputs.append(input_set)
            
        return inputs
    
    def _get_examplar_caption(self, level_idx):
        caption = f'Index-{level_idx}'
        
        return  {'type': 'text',
                 'text': caption}
    
    # Using image url
    def _get_image_prompt(self, asset: Asset, criterion: str):
        input_imgs = []
        input_prompt = []
        
        for input_type in self.input_types[criterion]:
            
            image = asset.image[input_type]
            asset_path = osp.join(asset.path, f'{input_type}.png')

            image_prompt = [{'type': 'text',
                             'text': f"{input_type}: "},
                            {'type': 'image_url',
                             'image_url': {
                    "url": f"data:image/jpeg;base64,{self._encode_image(asset_path)}"
                }}]
            
            input_imgs.append(image)
            input_prompt.extend(image_prompt)

        return input_prompt, input_imgs
    
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')