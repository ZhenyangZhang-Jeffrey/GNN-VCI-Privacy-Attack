#!/bin/bash

# Task 1.2: Train attacker classifier on Marson dataset
# Requirements: 
#   - Load frozen VCI encoder (victim model)
#   - Train classifier to predict treatment T from hidden representation Z
#   - Do NOT allow gradients to modify VCI encoder

eval "$(conda shell.bash hook)"
conda activate VCI
export WANDB_MODE=offline
DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

echo "=========================================================="
echo "🔐 Training attacker classifier on Marson (frozen VCI encoder)"
echo "=========================================================="

python main_classifier.py \
    --name "3.1_marson_classifier" \
    --data_name gene \
    --data_path "$DATA/datasets/marson_prepped.h5ad" \
    --artifact_path "$DATA/artifact/classifier" \
    --checkpoint_vci "$DATA/artifact/saves/3.1_marson_baseline_2026.03.15_15:36:47/model_seed=None_epoch=99.pt" \
    --hparams "hparams/hparams_gene.json" \
    --device cuda:0 \
    --batch_size 2048 \
    --max_epochs 100 \
    --checkpoint_freq 20

echo "✅ Marson classifier training complete!"
