#!/bin/bash
# A script to run the eval lib multiple times
SAMPLES=20
for CID in {0..4}
do
    docker run \
    -d\
    --rm \
    --gpus all \
    -v /home/jacoblab/global_planner:/workspace \
    -v /home/jacoblab/global_planner_data:/root/data \
    pytorch/pytorch:1.7.1-cuda11.0-cudnn8-jupyter-ompl\
    python3 eval_model_car.py \
    --modelFolder=/root/data/car_robot \
    --start=$((1+CID*SAMPLES)) \
    --numEnv=$SAMPLES\
    --valDataFolder=/root/data/forest_car/val \
    --numPaths=10\
    --explore
done