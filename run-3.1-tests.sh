#!/bin/bash

# 1. Environment setup
eval "$(conda shell.bash hook)"
conda activate vci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

# 2. Core: Create 16-dimensional config file (ensure sensitivity analysis works)
# This fixes the issue where CLI parameters couldn't override JSON config
mkdir -p hparams
cat << 'HJSON' > hparams/hparams_gene_dim16.json
{
    "outcome_emb_dim": 256,
    "treatment_emb_dim": 64,
    "covariate_emb_dim": 16,
    "latent_dim": 16,
    "encoder_width": 128,
    "encoder_depth": 3,
    "decoder_width": 128,
    "decoder_depth": 3,
    "discriminator_width": 64,
    "discriminator_depth": 3,
    "generator_lr": 0.0003,
    "generator_wd": 4e-07,
    "discriminator_lr": 0.0003,
    "discriminator_wd": 4e-07,
    "discriminator_freq": 2,
    "opt_step_size": 450,
    "max_grad_norm": -1,
    "grad_skip_threshold": -1,
    "patience": 20
}
HJSON

# 3. Define generic run function
run_vci_exp() {
    local NAME=$1
    local OMEGA2=$2
    local HPARAM_FILE=$3
    local EXTRA=$4

    echo ">>> Starting experiment: $NAME (omega2=$OMEGA2, hparams=$HPARAM_FILE)"
    
    python main.py \
        --name "$NAME" \
        --data_name gene \
        --data_path "$DATA/datasets/marson_prepped.h5ad" \
        --artifact_path "$DATA/artifact" \
        --hparams "$HPARAM_FILE" \
        --device cuda:0 \
        --omega0 1.0 \
        --omega1 1.7 \
        --omega2 "$OMEGA2" \
        --dist_outcomes normal \
        --dist_mode match \
        --max_epochs 100 \
        --batch_size 256 \
        --checkpoint_freq 20 \
        --eval_mode classic \
        $EXTRA
}

run_vci_exp "3.1_marson_baseline" "0.1" "hparams/hparams_gene.json" ""
