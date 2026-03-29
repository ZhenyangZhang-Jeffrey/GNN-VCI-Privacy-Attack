#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate VCI

DATA=$(cd -P -- "$(dirname -- "$0")" && pwd -P)

PYARGS=""
PYARGS="$PYARGS --name celebA-HQ-test"
PYARGS="$PYARGS --data_name celebA"
PYARGS="$PYARGS --data_path $DATA/data/celebA-HQ"
PYARGS="$PYARGS --artifact_path $DATA/artifact"
PYARGS="$PYARGS --hparams hparams/hparams_celebA.json"
PYARGS="$PYARGS --device cuda:0"

PYARGS="$PYARGS --omega0 10.0"
PYARGS="$PYARGS --omega1 0.05"
PYARGS="$PYARGS --omega2 0.01"
PYARGS="$PYARGS --dist_outcomes bernoulli"
PYARGS="$PYARGS --dist_mode discriminate"
#PYARGS="$PYARGS --checkpoint_classifier /path/to/trained/classifier"

PYARGS="$PYARGS --max_epochs 1000"
PYARGS="$PYARGS --batch_size 64"

python main.py $PYARGS
