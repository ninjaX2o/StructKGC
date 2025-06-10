## Improving Knowledge Graph Completion with Structure-Aware Supervised Contrastive Learning

This Code repository hosts the datasets, code and scripts to run experiments found in the anonymous submission
"[Improving Knowledge Graph Completion with Structure-Aware Supervised Contrastive Learning]"to EMNLP 2024.



## Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15

All experiments are run with 2 A100 GPUs.

## How to Run

It involves 3 steps: dataset preprocessing, model training, and model evaluation.


For WN18RR and FB15k237 datasets, we use files from [KG-BERT](https://github.com/yao8839836/kg-bert).

### FB15k-237 dataset

Step 1, extract the path from dataset
```
bash scripts/extractpath.sh FB15k237
```
Step 2, preprocess the dataset
```
bash scripts/preprocess.sh FB15k237
```
Step 3, training the model and (optionally) specify the output directory (< 3 hours)
```
OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh
```

Step 4, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/fb15k237/model_last.mdl FB15k237
```

Feel free to change the output directory to any path you think appropriate.

### WN18RR dataset
Step 1, extract the path from dataset
```
bash scripts/extractpath.sh WN18RR
```
Step 2, preprocess the dataset
```
bash scripts/preprocess.sh WN18RR
```
Step 3, training the model and (optionally) specify the output directory (< 3 hours)
```
OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh
```

Step 4, evaluate a trained model
```
bash scripts/eval.sh ./checkpoint/wn18rr/model_last.mdl WN18RR

