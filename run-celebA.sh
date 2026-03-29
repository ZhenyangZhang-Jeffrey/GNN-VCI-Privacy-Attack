#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate /workspace/vci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name celebA-HQ-test"
PYARGS="$PYARGS --data_name celebA"
PYARGS="$PYARGS --data_path /dev/shm/celebA-HQ" 
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --hparams hparams/hparams_celebA.json"
PYARGS="$PYARGS --label_names Male,Smiling" 
PYARGS="$PYARGS --device cuda:0"

PYARGS="$PYARGS --omega0 10.0"
PYARGS="$PYARGS --omega1 0.05"
PYARGS="$PYARGS --omega2 0.01"
PYARGS="$PYARGS --dist_outcomes bernoulli"
PYARGS="$PYARGS --dist_mode discriminate"
#PYARGS="$PYARGS --checkpoint_classifier /path/to/trained/classifier"

PYARGS="$PYARGS --max_epochs 100"
PYARGS="$PYARGS --batch_size 64"
PYARGS="$PYARGS --checkpoint_freq 10"

python main.py $PYARGS
