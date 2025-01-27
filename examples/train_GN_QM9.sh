#!/bin/bash

dev=1 # list CUDA devices to use
logdir="logs" # log dir for storing checkpoints and metrics

mkdir -p $logdir
CUDA_VISIBLE_DEVICES=$dev python scripts/train.py --conf examples/graph-network.yaml --dataset QM9 --dataset-root ~/data/qm9 --dataset-arg energy_U0 --log-dir $logdir --redirect true &
echo "Starting training... (saving logs at $logdir/log)"
