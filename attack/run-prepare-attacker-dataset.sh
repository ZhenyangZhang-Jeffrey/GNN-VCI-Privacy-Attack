#!/bin/bash

# Task 1.3: Build Attacker Dataset
# 从冻结的 VCI Encoder 中提取隐变量 Z
# 构建攻击者数据集（Z, Y, Donor Indicator）

eval "$(conda shell.bash hook)"
conda activate VCI
export WANDB_MODE=offline
DATA=$(cd -P -- "$(dirname -- "$0")/../" && pwd -P)

echo "=========================================================="
echo "🎯 Task 1.3: Building Attacker Dataset from Frozen VCI"
echo "=========================================================="

python prepare_attacker_dataset.py \
    --checkpoint_vci "$DATA/artifact/saves/3.1_marson_baseline_2026.03.15_15:36:47/model_seed=None_epoch=99.pt" \
    --data_path "$DATA/datasets/marson_prepped.h5ad" \
    --data_name gene \
    --artifact_path "$DATA/artifact" \
    --output_filename "marson_attacker_dataset.pt" \
    --device cuda:0 \
    --batch_size 128

echo "✅ Attacker dataset preparation complete!"
