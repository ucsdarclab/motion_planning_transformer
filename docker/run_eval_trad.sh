#!/bin/bash
# A script to run the eval lib multiple times
SAMPLES=40
for CID in {0..24}
do
    docker run \
    -d \
    --rm \
    --gpus all \
    -v /home/jacoblab/global_planner:/workspace \
    -v /home/jacoblab/global_planner_data:/root/data \
    pytorch/pytorch:1.7.1-cuda11.0-cudnn8-jupyter-ompl\
    python3 eval_model_trad.py \
    --plannerType=informedrrtstar \
    --start=$((CID*SAMPLES)) \
    --samples=$SAMPLES \
    --valDataFolder=/root/data/maze4/val780\
    --numPaths=1
done