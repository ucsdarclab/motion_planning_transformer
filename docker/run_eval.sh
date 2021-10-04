#!/bin/bash
# A script to run the eval lib multiple times
SAMPLES=500
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
    --segmentType=mpt \
    --modelFolder=/root/data/model42 \
    --plannerType=rrtstar \
    --start=$((CID*SAMPLES)) \
    --numEnv=$SAMPLES\
    --valDataFolder=/root/data/forest/val \
    --epoch=149\
    --numPaths=1\
    --explore
done