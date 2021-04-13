#!/bin/bash

docker run -it \
	-p $1:8888 \
	-v ~/global_planner_data:/root/data \
	tensorflow/tensorflow:latest-jupyter