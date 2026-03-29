#!/bin/bash

# Task 1.4: Train Attacker MLP for Model Inversion Attack
# 在冻结的 VCI Encoder 提取的隐变量 Z 上训练攻击者网络
# 目标：学习映射 f_attack: Z → Ŷ，重构原始基因表达向量

eval "$(conda shell.bash hook)"
conda activate VCI
export WANDB_MODE=offline
DATA=$(cd -P -- "$(dirname -- "$0")/../" && pwd -P)

echo "=========================================================="
echo "🎯 Task 1.4: Training Attacker MLP"
echo "=========================================================="

python train_attacker.py \
    --attacker_dataset "$DATA/artifact/marson_attacker_dataset.pt" \
    --architecture default \
    --dropout_rate 0.1 \
    --output_activation None \
    --learning_rate 0.001 \
    --weight_decay 1e-5 \
    --batch_size 128 \
    --max_epochs 100 \
    --early_stopping_patience 15 \
    --validation_ratio 0.2 \
    --artifact_path "$DATA/artifact" \
    --exp_name "marson_attacker" \
    --device cuda:0

echo "✅ Attacker network training complete!"
