#!/bin/bash

docker run -it \
	-p $1:8888 \
	--shm-size='32g'\
	--gpus all\
	-v ~/global_planner_data:/root/data \
	-v ~/global_planner:/workspace \
	pytorch/pytorch:1.7.1-cuda11.0-cudnn8-jupyter \
	bash