#!/bin/bash
DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
echo "🚀 Starting HAE (Hierarchical Autoencoder) training (100 Epochs)..."
/workspace/vci-env/bin/python main.py \
    --name "3.2_ablation_HAE" \
    --data_name morphoMNIST \
    --data_path "$DATA/datasets/morphoMNIST" \
    --artifact_path "$DATA/artifact" \
    --hparams "hparams/hparams_morphoMNIST.json" \
    --device cuda:0 \
    --omega0 10.0 \
    --omega1 0.00 \
    --omega2 0.00 \
    --dist_outcomes bernoulli \
    --dist_mode discriminate \
    --max_epochs 200 \
    --batch_size 512 \
    --checkpoint_freq 10
