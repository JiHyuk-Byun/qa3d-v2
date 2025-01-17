## Step 1. Set Environment
```
pip install -r requirements.txt
```

## Step 2. Prepare datasets
Prepare dataset and split .txt files
```
datasets
├── gobjaverse
│   ├── 0/11234
│   │   ├── albedo.png
│   │   ├── metallic_map.png
│   │   ├── mr.png
│   │   ├── normal_map.png
│   │   ├── rgb.png
│   │   └── roughness_map.png
│   │ ...
│   └── gobjaverse_id
│       ├── albedo.png
│       ├── metallic_map.png
│       ├── mr.png
│       ├── normal_map.png
│       ├── rgb.png
│       └── roughness_map.png
└── splits
    ├── 0.txt
    ├── 1.txt
    ├── ...
    └── n.txt
```
```
# inside 0.txt
0/10329
0/10668
0/11697
0/11857
0/12835
.
.
.
```
If you don' have merged multi-view images and only have gobajverse data, please run utils/preprocoess_gobjaverse.py
```
python utils/preprocess_gobjaverse.py -s PATH/TO/DATASET #e.g.) python utils/preprocess_gobjaverse.py -s datasets/gobjaverse_1000
```
## Step 3. Run
```
# RUN using qwen2-vl model
python main.py -c config/main_qwen2_vl.yaml
# RUN using gpt-4o model
python main.py -c config/main_gpt4o.yaml -u
```
