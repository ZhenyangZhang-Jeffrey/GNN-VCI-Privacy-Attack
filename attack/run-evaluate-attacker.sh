#!/bin/bash

# Task 1.5: Evaluate Attacker Model Against Three Scientific Objectives

eval "$(conda shell.bash hook)"
conda activate VCI
DATA=$(cd -P -- "$(dirname -- "$0")/../" && pwd -P)

echo "=========================================================="
echo "🧪 Task 1.5: Evaluating Attacker Model"
echo "=========================================================="

# 设置模型路径
ATTACKER_DATASET="$DATA/artifact/marson_attacker_dataset.pt"
ATTACKER_MODEL="$DATA/artifact/marson_attacker_*/attacker_model.pt"

# 如果没有指定具体模型，使用最新的
if [ ! -f "$ATTACKER_MODEL" ]; then
    echo "🔍 Finding the most recent attacker model..."
    ATTACKER_MODEL=$(find "$DATA/artifact" -name "attacker_model.pt" -type f | sort -r | head -1)
fi

if [ -z "$ATTACKER_MODEL" ] || [ ! -f "$ATTACKER_MODEL" ]; then
    echo "❌ Error: Attacker model not found!"
    echo "   Expected path: $DATA/artifact/marson_attacker_*/attacker_model.pt"
    exit 1
fi

echo "Using attacker model: $ATTACKER_MODEL"

python evaluate_attacker.py \
    --attacker_dataset "$ATTACKER_DATASET" \
    --model_checkpoint "$ATTACKER_MODEL" \
    --architecture default \
    --dropout_rate 0.1 \
    --output_activation None \
    --batch_size 128 \
    --device cuda:0 \
    --output_path "$DATA/artifact/attack_evaluation_results.json"

echo ""
echo "✅ Evaluation complete!"
echo "📊 Results saved to: $DATA/artifact/attack_evaluation_results.json"
