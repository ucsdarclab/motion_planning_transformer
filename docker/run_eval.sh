#!/bin/bash
# A script to run the eval lib multiple times
SAMPLES=125
for CID in {0..20}
do
    docker run \
    -d \
    --rm \
    --gpus all \
    -v /home/jacoblab/global_planner:/workspace \
    -v /home/jacoblab/global_planner_data:/root/data \
    pytorch/pytorch:1.7.1-cuda11.0-cudnn8-jupyter-ompl\
    python3 eval_model.py \
    --segmentType=unet \
    --modelFolder=/root/data/unet/model0 \
    --plannerType=rrtstar \
    --start=$((CID*SAMPLES)) \
    --samples=$SAMPLES\
    --valDataFolder=/root/data/forest/val \
    --epoch=69\
    --numPaths=1
done