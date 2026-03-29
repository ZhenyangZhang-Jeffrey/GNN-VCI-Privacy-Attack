#!/bin/bash

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

echo "=========================================================="
echo "⚖️ Silently training Morpho-MNIST property measurement model (without WandB)"
echo "=========================================================="

# Directly use Python absolute path in environment, run main_classifier.py
/workspace/vci-env/bin/python main_classifier.py \
    --name "3.2_mnist_classifier" \
    --data_name morphoMNIST \
    --label_names "thickness,intensity" \
    --data_path "$DATA/datasets/morphoMNIST" \
    --artifact_path "$DATA/artifact" \
    --hparams "hparams/hparams_morphoMNIST.json" \
    --device cuda:0 \
    --batch_size 256 \
    --max_epochs 100 \
    --checkpoint_freq 10

echo "✅ Referee model training complete!"
