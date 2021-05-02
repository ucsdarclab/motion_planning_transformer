#!/bin/bash
# A script to run the eval lib multiple times

for CID in {0..4}
do
    docker run \
    -d \
    --rm \
    --gpus all \
    -v /home/jacoblab/global_planner:/workspace \
    -v /home/jacoblab/global_planner_data:/root/data \
    pytorch/pytorch:1.7.1-cuda11.0-cudnn8-jupyter-ompl\
    python3 eval_model.py \
    --modelFolder=/root/data/model14 \
    --start=$((900+CID*20)) \
    --valDataFolder=/root/data/maze/val \
    --epoch=24\
    --numPaths=25
done