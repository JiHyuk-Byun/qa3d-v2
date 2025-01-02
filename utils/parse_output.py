import os
from os import path as osp
import re
import json

import glob
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser

CRITERION = ['geometry', 'texture', 'material', 'plausibility', 'artifacts', 'preference']
CONDITION = {'scoring': r"Score[^\d]*(\d+)$",
             'ordering': r"Index[^\d]*(\d+)$",    # ordering
             }   # scoring
parser = ArgumentParser()
parser.add_argument('--src_dir', '-s', type=str, default='outputs/2025-01-02/ordering/answers')
parser.add_argument('--prompt_type', '-t', type=str, default='ordering')
parser.add_argument('--n_choices', '-n', type=int, default=3)
args = parser.parse_args()

def _trim_score(analysis: str, prompt_type: str):
    
    analysis = analysis.strip().splitlines()

    last_line = analysis[-1].strip()
    match = re.search(CONDITION[prompt_type], last_line)
    
    if match:
        return int(match.group(1))
    else:
        return None

#median으로 grouping 하고, avg로 그 group안에서 sorting

def parse_outputs(result: dict, prompt_type: str, n_choices: int):
    '''
    - gid
    - score
      -criterion 
        - avg
        - median
      - .
      - .
      - .
    '''
    parsed_output = {'gobjaverse_index': "", 'score': {}}
    for criterion, answers in result.items():
        if criterion == "asset_gid":
            parsed_output["gobjaverse_index"] = answers # its gid
            continue
        
        concensus = {}
        score_lst = []
        score_iter = 0 
        cnt = 0
        
        for i, (_, analysis) in enumerate(answers.items()):
            
            score = _trim_score(analysis, prompt_type)
            if score: 
                score_iter += score
                cnt += 1
                if (i + 1) % n_choices == 0:
                    score_iter /= cnt
                    score_lst.append(score_iter)

                    score_iter = 0
                    cnt = 0
            else:
                continue
        concensus['scores'] = score_lst
        concensus['avg'] = np.mean(score_lst) if score_lst else None
        concensus['median'] = np.median(score_lst) if score_lst else None
        
        parsed_output['score'][criterion]=concensus

    return parsed_output


def main():
    
    src_dir = args.src_dir
    prompt_type = args.prompt_type
    n_choices = args.n_choices
    
    # Read gid list
    gid_lst = glob.glob(osp.join(src_dir, '*', '*/'))
    gid_lst = sorted(gid_lst, key=lambda x: int(x.split('/')[-3]+x.split('/')[-2]))
    result = []
    
    print('Number of gids to parse: ', len(gid_lst))
    for gid_path in tqdm(gid_lst):
        with open(osp.join(gid_path, 'output.json'), 'r') as f:
            data = json.load(f)
            
            parsed_data = parse_outputs(data, prompt_type=prompt_type, n_choices=n_choices)
    
        result.append(parsed_data)

    with open(osp.join(src_dir, 'parsed_scores.json'), 'w') as f:
        json.dump(result, f, indent=4)
    
    print('Complete!')
    

if __name__ == '__main__':
    main()