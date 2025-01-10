import os
from os import path as osp
import json


def get_oids(result_path):
    
    result_oids = []
    
    with open(osp.join(result_path), 'r') as f:
        results = json.load(f)
        
    for data in results:
        
        result_oids.append(data['metadata']['objaverse_index'])
        
    
    return result_oids

def filter_oids(result_oids, sample_oids):
    
    return [oid for oid in result_oids if oid not in sample_oids]


if __name__ == '__main__':
    
    result_path = 'outputs/2025-01-03/scoring_no-multi-choices/answers/parsed_scores2.json'
    sample_path = 'datasets/objaverse_samples.txt'

    with open(sample_path , 'r') as f:
        oids_lst = f.read().strip().split('\n')
        
    result_oids = get_oids(result_path)
    
    filtered_oids = filter_oids(result_oids=result_oids, sample_oids=oids_lst)
    
    print('\n'.join(filtered_oids))
    with open(osp.join(osp.dirname(result_path), 'not_existing_oids.txt'), 'w') as f:
        f.write('\n'.join(filtered_oids))
        
    