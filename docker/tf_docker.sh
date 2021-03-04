#!/bin/bash

docker run -it \
	-p $1:8888 \
	-v /home/jacoblab/global_planner_data:/root/data \
	tensorflow/tensorflow:latest-jupyter