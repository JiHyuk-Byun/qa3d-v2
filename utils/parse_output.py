import os
from os import path as osp
import re
import json
from argparse import ArgumentParser

import glob
from tqdm import tqdm
import numpy as np

CRITERIA = ['geometry', 'texture', 'material', 'plausibility', 'artifacts', 'preference']
CONDITION = {'scoring': r"Score[^\d]*(\d+)$",
             'ordering': r"Index[^\d]*(\d+)$",    # ordering
             }   # scoring
GID_TO_OID_PATH = 'assets/gobjaverse_index_to_objaverse.json'
EXAMPLE_META_PATH = 'assets/example_scores.json'

parser = ArgumentParser()
parser.add_argument('--src_dir', '-s', type=str, default='outputs/2025-01-02/ordering/answers')
parser.add_argument('--prompt_type', '-t', type=str, default='ordering')
parser.add_argument('--n_choices', '-n', type=int, default=3)
parser.add_argument('--num_level', '-l', type=int, default=5)

args = parser.parse_args()

def _from_order_to_score(order: int, criterion: str, gids: list):
        
    with open(EXAMPLE_META_PATH, 'r') as f:
        example_meta = json.load(f)
    
    # Convert gid into score 
    example_score = []
    for gid in gids:
        for example in example_meta:
            if example['metadata']['gobjaverse_index'] == gid:
                example_score.append(example['score'][criterion])
                
    if order == 0:
        return example_score[order]
    
    elif order == 5:
        return example_score[order-1]
    
    else: 
        return (example_score[order-1] + example_score[order])/2    
    
def _find_objaverse_index(gid: str, meta_path: str = GID_TO_OID_PATH):
    
    with open(meta_path, 'r') as f:
        data = json.load(f)
        
    return data[gid]


def _trim_score_and_analysis(analysis: str, prompt_type: str):
    
    analysis = analysis.strip()
    
    pattern = r"^(?:\*{0,2})?(?:#{0,3})?\s*Analysis\s*(?:\*{0,2})?\s*:?\s*(.*)$"
    analysis_match = re.match(pattern, analysis, flags=re.DOTALL)
    
    if analysis_match:
        analysis = analysis_match.group(1)
    else:
        return None, None

    last_line = analysis.splitlines()[-1].strip()
    score_match = re.search(CONDITION[prompt_type], last_line)
    
    if score_match:
        return analysis, int(score_match.group(1))
    else:
        return None, None

#median으로 grouping 하고, avg로 그 group안에서 sorting 
def parse_outputs(input_meta: dict, result: dict, prompt_type: str, n_choices: int, num_level: int):
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
    parsed_output = {'metadata': {}}
    for criterion, answers in result.items():
        if criterion == "asset_gid":
            parsed_output['metadata']["gobjaverse_index"] = answers # its gid
            parsed_output['metadata']['objaverse_index'] = _find_objaverse_index(answers)
            continue

        examplar_info = list(input_meta[criterion].values())
        concensus = {}
        score_lst = []
        score_iter = 0 
        cnt = 0
        
        analysis_lst = []
        for i, (_, _analysis) in enumerate(answers.items()):
            
            examplars = examplar_info[i * num_level:i * num_level + num_level]
            analysis, score = _trim_score_and_analysis(_analysis, prompt_type)

            if analysis != None and score != None:
                if prompt_type == 'ordering':
                    score = _from_order_to_score(order=score, criterion=criterion, gids=examplars)
                score_iter += score
                cnt += 1

                if (i + 1) % n_choices == 0:
                    score_iter /= cnt

                    score_lst.append(score_iter)
                    analysis_lst.append(analysis)
                    score_iter = 0
                    cnt = 0
            else:
                continue
            
        concensus['analysis'] = analysis_lst
        concensus['score'] = score_lst
        concensus['mean_score'] = np.mean(score_lst) if score_lst else None
        concensus['median_score'] = np.median(score_lst) if score_lst else None
        
        parsed_output[criterion]=concensus

    return parsed_output


def main():
    
    src_dir = args.src_dir
    prompt_type = args.prompt_type
    n_choices = args.n_choices
    num_level = args.num_level
    
    # Read gid list
    gid_lst = glob.glob(osp.join(src_dir, '*', '*/'))
    print(gid_lst[0])
    gid_lst = sorted(gid_lst, key=lambda x: int(x.split('/')[-3]+x.split('/')[-2]))
    result = []
    
    print('Number of gids to parse: ', len(gid_lst))
    for gid_path in tqdm(gid_lst):

        with open(osp.join(gid_path, 'input.json'), 'r') as f:
            input_meta = json.load(f)
        with open(osp.join(gid_path, 'output.json'), 'r') as f:
            output_meta = json.load(f)
            
        parsed_data = parse_outputs(input_meta, output_meta, prompt_type=prompt_type, n_choices=n_choices, num_level=num_level)
    
        result.append(parsed_data)

    # print Distribution
    scores = {k:[] for k in CRITERIA}
    for asset in result:
        for criterion in CRITERIA:
            if asset[criterion]['mean_score'] == None:
                continue
            scores[criterion].append(asset[criterion]['mean_score'])
    
    for criterion in CRITERIA:
        print(f"\nCriterion: {criterion}")
        print(f"- mean: {np.mean(scores[criterion]):.2f}")
        print(f"- std: {np.std(scores[criterion]):.2f}")
        print(f"- min: {np.min(scores[criterion]):.2f}")
        print(f"- max: {np.max(scores[criterion]):.2f}")
        print(f"- median: {np.median(scores[criterion]):.2f}")
        
    with open(osp.join(src_dir, 'parsed_scores2.json'), 'w') as f:
        json.dump(result, f, indent=4)
    
    print('Complete!')
    

if __name__ == '__main__':
    main()