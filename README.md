# Ligo_modified

This project is mostly based on https://github.com/VITA-Group/LiGO/tree/main. We modified the code accordingly to fit our methods. 


### Steps to run the code:
Be aware that this will take time to run.


**1. Download Data**

Get Bert training data:
```
bash data/wiki/get_data_cased.bash en
```

**2. Tokenize Data**

bash tokenization/tokenize_wiki_bert.bash


### Training BERT from Scratch

(6L, 512H) BERT

```
python run_lm_distributed.py --config configs/bert_wiki.txt --config_name configs/bert-6L-512H.json --output_dir <output_path> --max_steps 400000 --warmup_steps 10000 --should_continue
```

(12L, 768H) BERT

```
python run_lm_distributed.py --config configs/bert_wiki.txt --config_name configs/bert-12L-768H.json --output_dir <output_path> --max_steps 400000 --warmup_steps 10000 --should_continue
```

### Training BERT with LiGO modified

First train a LiGO operator using the following command:

```
python run_grow_distributed.py --config configs/bert_wiki.txt --config_name configs/bert-12L-768H.json --output_dir <path_to_save_LiGO> --tune_width --tune_depth --source_model_path <path_to_small_model> --fuse_init_scheme stackbert_noisy rand --max_steps 100 --logging_steps 100 --ckpt_steps 100 --should_continue
```

Then use pre-trained LiGO operator to grow the model:

```
python run_lm_distributed.py --config configs/bert_wiki.txt --config_name configs/bert-12L-768H.json --output_dir <output_path> --grow_scheme ligo --source_model_path <path_to_small_model>  --pretrained_ligo_path <path_to_save_LiGO> --fuse_init_scheme stackbert_noisy rand --learning_rate 2e-4 --warmup_steps 0 --should_continue
```
### Convex hull experiments
The experiments to assess the viability of the convex hull are documented in test.ipybn
