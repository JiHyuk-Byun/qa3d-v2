import json




def main():
    
    json_path = "assets/example_scores.json"
    
    gid_lst = []
    
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    for metadata in data:
        gid_lst.append(metadata['metadata']['gobjaverse_index'])
        
    
    with open('assets/example.json', 'w') as f:
        json.dump(gid_lst, f, indent=4)
        


if __name__ == '__main__':
    main()
    
    print('Clear!')