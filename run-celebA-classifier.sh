#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate /workspace/vci-env

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name celebA-classifier"
PYARGS="$PYARGS --data_name celebA"
PYARGS="$PYARGS --data_path /dev/shm/celebA-HQ"
PYARGS="$PYARGS --artifact_path $DATA/artifact/classifier"
PYARGS="$PYARGS --hparams hparams/hparams_celebA.json"
PYARGS="$PYARGS --label_names Male,Smiling" 
PYARGS="$PYARGS --device cuda:0"

PYARGS="$PYARGS --max_epochs 50"
PYARGS="$PYARGS --batch_size 64"

python main_classifier.py $PYARGS
